import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer


class PDFSearch:
    def __init__(self):
        self.file_path = "multimodal/data/article_2502.15214v1.pdf"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.text = self.load_pdf()
        self.chunks = self.chunk_text()
        self.embeddings = self.calc_embeddings()
        self.index = self.faiss_index()

    def load_pdf(self):
        with open(self.file_path, "rb") as file:
            pdf = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(pdf.getNumPages()):
                text += pdf.getPage(page_num).extract_text()
        return text
    
    def chunk_text(self, chunk_size=1500): #chose to have 1,5k char chunck size
        chunks = []
        for i in range(0, len(self.text), chunk_size):
            chunks.append(self.text[i:i + chunk_size])
        return chunks

    def calc_embeddings(self):
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.chunks)
        return embeddings

    def faiss_index(self):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    def search_query(self, query):
        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode([query])
        distance, indices = self.index.search(query_embedding, k=2) #chose to keep the top 2 closest text extracts
        return [self.embeddings[i] for i in indices[0]] , indices[0], distance[0]

if __name__ == '__main__':
    module = PDFSearch()
    query = "What is the difference between LLM and VLM?"
    _ , indices, distance = module.search_query(query)
    reversed_sentences = [module.chunks[i] for i in indices]
    print("Calculated closest text extracts:", reversed_sentences)
    print("With distances:", distance)