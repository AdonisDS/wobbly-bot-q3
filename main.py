import argparse
import asyncio
import logging
from datetime import datetime, date
from typing import Literal, List, Sequence

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    ForecastReport,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class WobblyBot2025Q3(ForecastBot):
    _max_concurrent_questions = (5)
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}

                Your most important job is to search polymarket.com, then kalshi.com, then metaculus.com, then manifold.markets for their predictions and report their predictions as percentages. If a question exists on those websites, the percentages there will always be available and you should report their value.
                
                You'll always be able to find the same question on metaculus.com because you're being given a question from there, so keep trying until you find it.
                
                If it's a question about a sporting event, also search betting markets such as betfair.com and oddschecker.com and calculate the implied probabilities form the odds.
                
                Complement by searching and reporting on other websites that could have predictions by superforecasters or other type of professional forecasts and give the values of their predictions.
                                
                Then, if applicable for this question, search and report the base rates. Your should inform how many times the event happened in the last (up to) 40 years and what the period considered was. 
                
                After that, report on the current status of the situation as of the current date.

                Finally, generate a concise but detailed rundown of the most relevant recent news, including if the question would resolve Yes or No based on current information. Report as many news as possible.
                
                For each task and for each one of the 6 websites listed before, report their predictions separately or report that you couldn't find anything there or that they're not applicable for the question.

                Explain how you're following each of those steps.

                Finish by delivering the full final report. There won't be another follow-up prompt, so do as much as asked now.

                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.  Think very hard before answering

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            Your research assistant says:
            {research}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            The community prediction currently is:
            {question.community_prediction_at_access_time}
            Your prediction should differ from the community prediction by a Jeffreys Divergence of no more than 20%

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
           
            If the research was able to find probabilities from prediction markets, your prediction should mostly be based on that, with few adjustements to account for recent news.

            If data from prediction markets was not available but the community prediction was, make your prediction based on that with a few adjustements to account for the recent news.

            If all of that is unavailable, make your prediction grounded on the base rates, if available, and your independent rationale.

            If the event being forecasted can happen any time, then assume that the probability of the event happening decays linearly over time, so that if not a lot of time if left until resolution, you would give a very low probability. To do that, first calculate the probability that the event happens in the next 12 months according to the base rates, and then adjust that probability based on the time left until resolution This is not applicable for events that only happen once at a specified date, such as an election.

            Good forecasters also leave room for uncertainty even possibilities outside the model and what showed up in the research, so never predict anything below 2% ou above 98%

            Explain how you're following each of those steps.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)
    
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        #TODO Make the bot skip questions already predicted today, like it was done for the binary questions in test_questions mode
        reports: list[ForecastReport | BaseException] = []
        reports = await asyncio.gather(
            *[
                self._run_individual_question_with_error_propagation(question)
                for question in questions
            ],
            return_exceptions=return_exceptions,
        )

        return reports

    def make_default_binary_prediction(self):
        return 0.5
    
    def make_default_numeric_prediction(self, question: NumericQuestion):
        if (question.open_lower_bound and question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0.05, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=0.95, value=question.upper_bound),
            ]
        elif (question.open_lower_bound and not question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0.05, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=1, value=question.upper_bound),
            ]
        elif (not question.open_lower_bound and question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=0.95, value=question.upper_bound),
            ]
        else:
            percentile_list: List[Percentile] = [
                Percentile(percentile=0, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=1, value=question.upper_bound),
            ]
        return NumericDistribution.from_question(percentile_list, question)
    
    def make_default_multiple_choice_prediction(self, question: MultipleChoiceQuestion):
        num_options = len(question.options)
        probability_per_option = 1.0 / num_options
        probabilities: dict[str, float] = dict.fromkeys(question.options, probability_per_option)
        return probabilities
    
    @staticmethod
    def load_data_from_file(filepath) -> dict:
        data = {}
        try:
            with open(filepath, "r") as f:
                for line in f:
                    # Skip empty lines
                    if line.strip():
                        # Split only on the first colon to handle values that might contain colons
                        key, value = line.strip().split(":", 1)
                        data[key] = value
        except FileNotFoundError:
            # If the file doesn't exist, just return an empty dictionary
            logger.error(f"'{filepath}' not found. Starting with an empty dataset.")
        return data
    
    @staticmethod
    def save_data_to_file(data: dict, filepath):
        with open(filepath, "w") as f:
            for key, value in data.items():
                f.write(f"{key}:{value}\n")
        print(f"Successfully saved data to '{filepath}'.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the WobblyBot2025Q3"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aib_tournament", "metaculus_cup", "mini_bench", "test_questions"],
        default="test_questions",
        help="Specify the run mode (default: test_questions)",
    )
    args = parser.parse_args()
    run_mode: Literal["aib_tournament", "metaculus_cup", "mini_bench", "test_questions"] = args.mode
    assert run_mode in [
        "aib_tournament",
        "metaculus_cup",
        "mini_bench",
        "test_questions",
    ], "Invalid run mode"

    bot = WobblyBot2025Q3(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        enable_summarize_research=False,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={  
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "forecaster": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "researcher": GeneralLlm(
                model="openrouter/openai/gpt-5:online",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            )
        }
    )

if run_mode == "aib_tournament":
    logger.info("Running Wobbly Bot in AIB tournament mode")

    questions = MetaculusApi.get_all_open_questions_from_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID)
    asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

elif run_mode == "metaculus_cup":
    logger.info("Metaculus Cup mode not implemented yet") #TODO
elif run_mode == "mini_bench":
    logger.info("Mini Bench mode not implemented yet") #TODO
elif run_mode == "test_questions":
    logger.info("Running Wobbly Bot in test mode")

    EXAMPLE_QUESTIONS = [
        #578: Human Extinction - Binary
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  
        #8632: Total Yield of Nuc Det 1000MT by 2050 - Binary
        "https://www.metaculus.com/questions/8632/total-yield-of-nuc-det-1000mt-by-2050/",
        #14333: Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        #22427: Number of New Leading AI Labs - Multiple Choice  
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        #38880: Number of US Labor Strikes Due to AI in 2029 - Discrete  
        "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
    ]

    prediction_date_dict = bot.load_data_from_file("latest_prediction_dates_test_questions.txt")
    today = date.today().isoformat()

    for question_url in EXAMPLE_QUESTIONS:
        question = MetaculusApi.get_question_by_url(question_url)
        if question.question_type == "binary":
            if question.already_forecasted:
                if (today == prediction_date_dict.get(str(question.id_of_question))):
                    logger.info("Already made a prediction today on question " + str(question.id_of_question) + ": " + question.question_text)
                    continue
                logger.info("Updating the prediction on question " + str(question.id_of_question) + ": " + question.question_text)
                MetaculusApi.post_binary_question_prediction(question.id_of_question,bot.make_default_binary_prediction())
                prediction_date_dict[str(question.id_of_question)] = today
            else:
                logger.info("Making the first prediction on question " + str(question.id_of_question) + ": " + question.question_text)
                MetaculusApi.post_binary_question_prediction(question.id_of_question,bot.make_default_binary_prediction())
                prediction_date_dict[str(question.id_of_question)] = today
         
        elif question.question_type == "numeric":
            prediction = bot.make_default_numeric_prediction(question)
            cdf = [percentile.percentile for percentile in prediction.cdf]
            MetaculusApi.post_numeric_question_prediction(question.id_of_question, cdf)
        elif question.question_type == "multiple_choice":
            MetaculusApi.post_multiple_choice_question_prediction(question.id_of_question, bot.make_default_multiple_choice_prediction(question))
        elif question.question_type == "date":
            continue # As of August 2025, this question type is still not supported by Metaculus for bots
        elif question.question_type == "discrete":
            #TODO make the code for disctrete questions more specialized
            prediction = bot.make_default_numeric_prediction(question)
            cdf = [percentile.percentile for percentile in prediction.cdf]
            MetaculusApi.post_numeric_question_prediction(question.id_of_question, cdf)        
    bot.save_data_to_file(prediction_date_dict, "latest_prediction_dates_test_questions.txt")