from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class LocalInference:
    def __init__(self):
        self.blip = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.blip)
        self.model = BlipForConditionalGeneration.from_pretrained(self.blip)

    def infer(self, raw_image, text=None):
        inputs = self.processor(raw_image, text, return_tensors="pt")
        out = self.model.generate(**inputs)
        print(self.processor.decode(out[0], skip_special_tokens=True))

    def main(self):
        raw_image = Image.open("multimodal/data/demo.jpg").convert('RGB')

        #unconditional image captioning
        self.infer(raw_image)

        #conditional image captioning: add a text prompt that the LLM has to include in the answer
        self.infer(raw_image, text="a photography of")

if __name__ == '__main__':
    module = LocalInference()
    module.main()