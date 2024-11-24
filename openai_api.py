from openai import OpenAI



class OpenAIAPI:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def create_chat_completion(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content
