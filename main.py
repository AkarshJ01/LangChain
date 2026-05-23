import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain!")

    information = """
                Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor,[2] and investor. A pioneer of the personal computer revolution of the 1970s and 1980s, Jobs co-founded Apple Inc. with his early business partner Steve Wozniak as Apple Computer Company in 1976. After the company's board of directors fired him in 1985, he founded NeXT the same year and purchased Pixar in 1986, becoming its chairman and majority shareholder until 2007. Jobs returned to Apple in 1997 as CEO, where he was closely involved with the creation and promotion of many of the company's most influential products until his resignation in 2011.

                Jobs was born in San Francisco in 1955 and adopted shortly afterward. He attended Reed College in 1972 before withdrawing that same year. In 1974, he traveled through India, seeking enlightenment before later studying Zen Buddhism. He and Wozniak co-founded Apple in 1976 to further develop and sell Wozniak's Apple I personal computer. Together, the duo gained fame and wealth a year later with the production and sale of the Apple II, one of the first highly successful mass-produced microcomputers.
                """


    summary_template = """
    Given the information {information} create : 
    1. A short summary 
    2. 2 Interesting facts about them
"""

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )

    llm = ChatOllama(temperature = 0, model = "gemma4:e4b")
    chain = summary_prompt_template | llm

    response = chain.invoke(input = {"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
