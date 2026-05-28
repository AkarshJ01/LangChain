from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model
from langchain.tools import tools
from langchain_core.messages import HumanMessage, SystemMessage , ToolMessage

MAX_ITR = 10

MODEL = "llama3.1:8b"

@tools
def get_product_price(product : str) -> float:
    """Finding price of the product"""

    print(f"Product being found : {product}")

    price = {"laptop" : 1299 , "headphones" : 699 , "keyboard" : 15.9}

    return price.get(product , 0)


@tools
def apply_discount(price: float , discount_tier: str) -> float:
    """Outputs the final price after discount"""
    print (f"discount tier : {discount_tier}")
    discount_percentage = {"bronze" : 10, "silver" : 20, "gold" : 30}
    dis = discount_percentage.get(discount_percentage , 0)
    price = price - price * dis /100
    return price

# --- Agent Loop ---

def run_agent(question: str):
    pass



if __name__ == "__main__":
    print(f"Langchain model : {MODEL} \n")
 
