import requests
from multimodal.utils.client import LLMClient
from multimodal.utils.load_pdf import PDFSearch

class ApiCalls:
    def __init__(self):
        self.client = LLMClient() #authenticate the HF client
        self.pdf = PDFSearch() #pre-load the embeddings of the pdf
        #### Models used:
        self.blip = "Salesforce/blip-image-captioning-base"
        self.ner = "dslim/bert-base-NER"
        self.mistral = "mistralai/Mistral-7B-Instruct-v0.3"

    def entity_recognition(self, text: str) -> str:
        """
        This function uses the token_classification() function from the HF api to generate 
        entities (place nemes, people names...) from a given text.
        """
        result = self.client.token_classification(
            text,
            model=self.ner,
        )

        return f"NER model found that: {result[0].word} is a {result[0].entity_group}"

    def img2text(self, img_path: str) -> str:
        """
        This function uses the image_to_text() function from the HF api to generate a text from an image.
        """
        if img_path.startswith("http"):
            img_data = requests.get(img_path).content
        else:
            img_data = img_path
        output = self.client.image_to_text(img_data, model=self.blip)

        return output.generated_text

    def RAG_query(self, question: str) -> str:
        """
        This function retrieves the relevant information from a PDF document and adds it to
        the prompt of the model to simmulate a RAG method.
        """
        context = self.pdf.search_query(question)
        output = self.client.chat.completions.create(
            model=self.mistral,
            messages=[
                {
                    "role": "system",
                    "content": "You are a bright research assistant. Answer the following question using the context provided."
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}

                    Context: {context}"""
                }
            ],
            max_tokens=300,
        )

        return output.choices[0].message.content
    
    def summarize_and_key_points(self, text: str) -> str:
        """
        This function asks the LLM to respond to a user query by summarizing a given text and providing
        three key points.
        """
        output = self.client.chat.completions.create(
            model=self.mistral,
            messages=[
                {
                    "role": "system",
                    "content": "You have a brilliant mind, you are able to summarize complex ideas or long texts very gracefully."
                },
                {
                    "role": "user",
                    "content": f"""Summarize the following text and provide three key points in the shape of bullet points: 
                    {text}"""
                }
            ],
            max_tokens=300,
        )

        return output.choices[0].message.content

    def main(self):
        #1. Summarize and key points 3k char from the pdf
        print(self.summarize_and_key_points(self.pdf.text[0:5000]))

        #2. Image to text
        img_path = "multimodal/data/demo.jpg"
        print(self.img2text(img_path))

        #3. Image to text from URL with a summarization and key points
        img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
        response = self.img2text(img_url)
        print(self.summarize_and_key_points(response))

        #4. RAG query
        precise_question = "What is long-horizon planning, and reward design?"
        print((self.RAG_query(precise_question)))

        #5. Entity recognition
        text = "My name is Sarah Jessica Parker but you can call me Jessica"
        print(self.entity_recognition(text))

if __name__ == '__main__':
    module = ApiCalls()
    module.main()