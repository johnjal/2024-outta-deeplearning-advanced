import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

# [1] 빈칸을 작성하시오.
# API 키
GOOGLE_API_KEY = "AIzaSyAD-m2T-seSya1HtM6wT93cnsaCuQX39Ss"
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 초기화
def LLM(text):
  model = genai.GenerativeModel('gemini-pro')
  
  while True:
    user_input = text
    if user_input=="q":
      break
    else:
      """
      [Example]
      User: I want to change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
      You: {"top":["blue","sweatshirt"],"bottom":["black","skirt"]}
      """
      # [2] 빈칸을 작성하시오.
      # 예시와 같이 성능 향상을 위해 프롬프트 튜닝을 진행
      instruction = """Your answer is used directly to code execution. Answers are needed to be accurate and formatted. 
      Input is a sentence about changing clothes. If the sentence only about top cloth, you need to output '0' at the front and description of the changed cloth at the back.
      If the sentence only about bottom cloth, you need to output '1' at the front and description of the changed cloth at the back.
      If the sentence about top and bottom cloth, you need to output '2' at the front and description of the changed cloth at the back.
      You need to connect number and the description with '+'.
      """
      prompt = """
      Change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
      A: 2+blue sweatshirt, black skirt

      Change the woman's dress to a black cocktail dress.
      A: 2+black cocktail dress

      Replace the man's shirt with a navy blue polo shirt.
      A: 0+navy blue polo shirt

      Switch the woman's skirt to a red pleated skirt.
      A: 1+red pleated skirt

      Change the girl's dress to a floral summer dress.
      A: 2+floral summer dress

      Replace the man's jacket with a leather bomber jacket.
      A: 0+leather bomber jacket

      Switch the boy's shorts to cargo shorts.
      A: 1+cargo shorts

      Change the woman's top to a white silk blouse.
      A: 0+white silk blouse

      Replace the girl's jeans with ripped skinny jeans.
      A: 1+ripped skinny jeans

      Change the man's suit to a black tuxedo.
      A: 2+black tuxedo

      Switch the woman's trousers to wide-leg pants.
      A: 1+wide-leg pants

      Change the girl's hoodie to a purple zip-up hoodie.
      A: 0+purple zip-up hoodie

      Replace the man's coat with a trench coat.
      A: 0+trench coat

      Switch the woman's leggings to yoga pants.
      A: 1+yoga pants

      Change the boy's top to a green sweater.
      A: 0+green sweater

      Replace the woman's jumpsuit with a denim overalls.
      A: 2+denim overalls

      Change the man's jeans to straight-leg jeans.
      A: 1+straight-leg jeans

      Switch the girl's top to a striped long-sleeve shirt.
      A: 0+striped long-sleeve shirt

      Change the woman's skirt to a pencil skirt.
      A: 1+pencil skirt

      Replace the man's trousers with black dress pants.
      A: 1+black dress pants

      Switch the woman's blouse to a red silk blouse.
      A: 0+red silk blouse

      Change the boy's shorts to denim shorts.
      A: 1+denim shorts

      Replace the woman's dress with a navy evening gown.
      A: 2+navy evening gown

      Switch the girl's sweater to a pink pullover.
      A: 0+pink pullover

      Change the man's coat to a wool overcoat.
      A: 0+wool overcoat

      Replace the woman's jeans with high-waisted jeans.
      A: 1+high-waisted jeans

      Switch the boy's jacket to a puffer jacket.
      A: 0+puffer jacket

      Change the woman's shorts to linen shorts.
      A: 1+linen shorts

      Replace the man's hoodie with a grey zip-up hoodie.
      A: 0+grey zip-up hoodie

      Switch the girl's dress to a lace party dress.
      A: 2+lace party dress

      """

      # [3] 빈칸을 작성하시오.
      # 전체 프롬프트 생성 (instruction 포함)
      full_prompt = f"{instruction}\n{prompt}\n{user_input}"

      # [4] 빈칸을 작성하시오.
      # 모델 호출
      response = model.generate_content(full_prompt)

      # 응답 출력
      return response