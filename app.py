import streamlit as st
import boto3
import time
import config
import langchain
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Page configuration

st.set_page_config(
    page_title='AWS Glue Data Catalog Text-to-SQL',
    page_icon=':space_invader:',
    initial_sidebar_state='collapsed')
st.title(':violet[AWS Glue] Data Catalog Text-to-SQL :space_invader:')
st.caption('Supercharge your Glue Data Catalog :rocket:')

# Variables

langchain.verbose = True
session = boto3.session.Session()
region = config._global['region']
credentials = session.get_credentials()
service = 'es'
http_auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token)
opensearch_cluster_domain_endpoint = config.opensearch['domain_endpoint']
domain_name = config.opensearch['domain_name']
index_name = "index-superglue"

# Create AWS Glue client

glue_client = boto3.client('glue', region_name=region)

# Function to get all tables from Glue Data Catalog


def get_tables(glue_client):
    # get all AWS Glue databases
    databases = glue_client.get_databases()

    tables = []

    num_db = len(databases['DatabaseList'])

    for db in databases['DatabaseList']:
        tables = tables + \
            glue_client.get_tables(DatabaseName=db['Name'])["TableList"]

    num_tables = len(tables)

    return tables, num_db, num_tables

# Function to flatten JSON representations of Glue tables


def dict_to_multiline_string(d):

    lines = []
    db_name = d['DatabaseName']
    table_name = d['Name']
    columns = [c['Name'] for c in d['StorageDescriptor']['Columns']]

    line = f"{db_name}.{table_name} ({', '.join(columns)})"
    lines.append(line)

    return "\n".join(lines)

# Function to render user input elements


def render_form(catalog):
    if (num_tables or num_db):
        st.write(
            "A total of ",
            num_tables,
            "tables and ",
            num_db,
            "databases were indexed")

    k = st.selectbox(
        'How many tables do you want to include in table search result?',
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        index=2)

    query = st.text_area(
        'Prompt',
        "What is the total inventory per warehouse?")

    with st.sidebar:
        st.subheader(":violet[Data Catalog] :point_down:")
        st.write(catalog)

    return k, query

# Function to perform a similarity search


def search_tables(vectorstore, k, query):
    relevant_documents = vectorstore.similarity_search_with_score(query, k=k)
    for rel_doc in relevant_documents:
        st.write(rel_doc[0].page_content.split(" ")[0])
        st.write("Score: ", rel_doc[1])
        st.divider()


# Function to generate LLM response (SQL + Explanation)

def generate_sql(vectorstore, k, query):
    prompt_template = """
     \n\nHuman: Between <context></context> tags, you have a description of tables with their associated columns. Create a SQL query to answer the question between <question></question> tags only using the tables described between the <context></context> tags. If you cannot find the solution with the provided tables, say that you are unable to generate the SQL query.

    <context>
    {context}
    </context>

    Question: <question>{question}</question>

    Provide your answer using the following xml format: <result><sql>SQL query</sql><explanation>Explain clearly your approach, what the query does, and its syntax</explanation></result>

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=bedrock_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )
    with st.status("Generating response :thinking_face:"):
        answer = qa({"query": query})

    # st.write(answer)

    with st.status("Searching tables :books:"):
        time.sleep(1)

    for i, rel_doc in enumerate(answer["source_documents"]):
        st.write(rel_doc.page_content.split(" ")[0])

    with st.status("Rendering response :fire:"):
        sql_query = answer["result"].split("<sql>")[1].split("</sql>")[0]
        explanation = answer["result"].split("<explanation>")[
            1].split("</explanation>")[0]

    st.code(sql_query, language='sql')
    st.link_button(
        "Athena console :sun_with_face:",
        "https://{0}.console.aws.amazon.com/athena/home?region={0}".format(region))

    st.write(explanation)

# Amazon Bedrock LangChain clients


bedrock_llm = Bedrock(
    model_id="anthropic.claude-v2",
    model_kwargs={
        'max_tokens_to_sample': 3000})
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# VectorDB type

vectorDB = st.selectbox(
    "VectorDB",
    ("FAISS (local)", "OpenSearch (Persistent)"),
    index=0
)

if vectorDB == "FAISS (local)":

    with st.status("Connecting to Glue Data Catalog :man_dancing:"):

        catalog, num_db, num_tables = get_tables(glue_client)

        # Check if an index copy of FAISS is stored locally

        try:
            vectorstore_faiss = FAISS.load_local(
                "faiss_index", bedrock_embeddings)
        except BaseException:
            docs = [
                Document(
                    page_content=dict_to_multiline_string(x),
                    metadata={
                        "source": "local"}) for x in catalog]

            vectorstore_faiss = FAISS.from_documents(
                docs,
                bedrock_embeddings,
            )

            vectorstore_faiss.save_local("faiss_index")

    k, query = render_form(catalog)

    if st.button('Search relevant tables :dart:'):

        search_tables(vectorstore=vectorstore_faiss, k=k, query=query)

    if st.button('Generate SQL :crystal_ball:'):

        generate_sql(vectorstore=vectorstore_faiss, k=k, query=query)

elif vectorDB == "OpenSearch (Persistent)":

    with st.status("Connecting to Glue Data Catalog :man_dancing:"):

        catalog, num_db, num_tables = get_tables(glue_client)

        # Initialize Opensearch Vector Search clients

        vectorstore_opensearch = OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=bedrock_embeddings,
            opensearch_url=opensearch_cluster_domain_endpoint,
            engine="faiss",
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            http_auth=http_auth,
            connection_class=RequestsHttpConnection
        )

    k, query = render_form(catalog)

    if st.button('Search relevant tables :dart:'):
        search_tables(vectorstore=vectorstore_opensearch, k=k, query=query)

    if st.button('Generate SQL :crystal_ball:'):

        generate_sql(vectorstore=vectorstore_opensearch, k=k, query=query)
