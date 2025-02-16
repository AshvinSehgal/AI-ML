from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import pacmap
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm.notebook import tqdm
import os


os.environ["TOKENIZERS_PARALLELISM"] = 'false'

# Step 1: Load documents
dataset = load_dataset('m-ric/huggingface_doc', split='train')
document = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(dataset)
]
print('Created Documents...')

# Step 2: Split documents into chunks
separators = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'),
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=separators
)
docs = []
for doc in document:
    docs += text_splitter.split_documents([doc])
unique_texts = {}
docs_unique = []
for doc in docs:
    if doc.page_content not in unique_texts:
        unique_texts[doc.page_content] = True
        docs_unique.append(doc)
print('Created chunks...')

# Step 3: Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    multi_process=True,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store = FAISS.from_documents(docs, embedding_model)
retriever = vector_store.as_retriever()
print('Created embeddings...')

user_query = "What is RAG?"
query_vector = embedding_model.embed_query(user_query)

embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

embeddings_2d = [
    list(vector_store.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs))
] + [query_vector]

# Fit the data (the index of transformed data corresponds to the index of the original data)
documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

df = pd.DataFrame.from_dict(
    [
        {
            "x": documents_projected[i, 0],
            "y": documents_projected[i, 1],
            "source": docs[i].metadata["source"].split("/")[1],
            "extract": docs[i].page_content[:100] + "...",
            "symbol": "circle",
            "size_col": 4,
        }
        for i in range(len(docs))
    ]
    + [
        {
            "x": documents_projected[-1, 0],
            "y": documents_projected[-1, 1],
            "source": "User query",
            "extract": user_query,
            "size_col": 100,
            "symbol": "star",
        }
    ]
)

print('Plotting embedding graph...')
# Visualize the embedding
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="source",
    hover_data="extract",
    size="size_col",
    symbol="symbol",
    color_discrete_map={"User query": "black"},
    width=1000,
    height=700,
)
fig.update_traces(
    marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
    selector=dict(mode="markers"),
)
fig.update_layout(
    legend_title_text="<b>Chunk source</b>",
    title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
)
fig.show()

# Step 4: Load HF model for generation
model_name = 'google/flan-t5-large'  # Change to any HF model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=text_generator)
print('LLM Loaded...')

# Step 5: Create Retrieval-Augmented Generation (RAG) pipeline
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

print('Testing sample query...')
# Step 6: Query the RAG system
# query = 'What is RAG?'
query = input('You:')

# print("FAISS index contains:", vector_store.index.ntotal, "documents")
# print("Running query:", query)

retrieved_docs = retriever.invoke(query)
print("Retrieved Documents:", [doc.page_content[:200] for doc in retrieved_docs])

response = qa_chain.invoke({'query': query})
print('Response:', response)