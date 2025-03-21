from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class LocalInference:
    def __init__(self):
        self.blip = "Salesforce/blip-image-captioning-base"

    def load_blip(self):
        processor = BlipProcessor.from_pretrained(self.blip)
        model = BlipForConditionalGeneration.from_pretrained(self.blip)
        return processor, model

    def infer(self, raw_image, processor, model, text=None):
        inputs = processor(raw_image, text, return_tensors="pt")
        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

    def main(self):
        processor, model = self.load_blip()
        raw_image = Image.open("multimodal/data/demo.jpg").convert('RGB')

        # conditional image captioning
        self.infer(raw_image,processor, model)

        # unconditional image captioning
        self.infer(raw_image,processor, model, text="a photography of")

if __name__ == '__main__':
    module = LocalInference()
    module.main()