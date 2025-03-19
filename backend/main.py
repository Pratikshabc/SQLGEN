# from fastapi import FastAPI, Request, HTTPException, UploadFile, File
# from fastapi import Depends, status
# from fastapi.security import OAuth2PasswordBearer
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_openai import AzureOpenAI
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from langchain.chains import create_sql_query_chain
# from langchain_community.utilities import SQLDatabase  # Updated import
# from langchain_community.cache import SQLiteCache  # Updated import
# from langchain import hub
# from langchain_openai import AzureChatOpenAI
# from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# from langchain_community.vectorstores import AzureSearch  # Updated import
# from langchain.globals import set_llm_cache
# import os
# import ast
# from dotenv import load_dotenv
# from typing import List
# import secrets
# import jwt
# from datetime import datetime, timedelta
# from pydantic import BaseModel
# from starlette.middleware.sessions import SessionMiddleware
# from starlette.requests import Request
# from azure.search.documents.indexes import SearchIndexClient
# from azure.core.credentials import AzureKeyCredential
# from pymongo import MongoClient
# import pathlib
 
# load_dotenv()
 
# app = FastAPI()
 
# class DocumentRequest(BaseModel):
#     question: str
#     filename: str
 
# class User(BaseModel):
#     username: str
#     password: str  
 
# origins = [
#     "http://localhost:5173",
#     "*"  # React's default port
#     # add any other origins that need to access the API
# ]
 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
 
# SECRET_KEY = secrets.token_hex(32)
# ALGORITHM = "HS256"
# app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)
 
# client = MongoClient("mongodb://localhost:27017/")
# db = client["user_db"]
# users = db["users"]
# sql_history = db["sql_history"]
# documents = db["documents"]
# documents_history = db["documents_history"]

# # Create data directory if it doesn't exist
# data_dir = pathlib.Path("data")
# data_dir.mkdir(exist_ok=True)

# # Check if Chinook.db exists, if not create an empty file
# db_path = data_dir / "Chinook.db"
# if not db_path.exists():
#     db_path.touch()
#     print(f"Created empty database file at {db_path}")

# # Use absolute path for SQLite database
# db_uri = f"sqlite:///{db_path.absolute()}"
 
# sql_db = SQLDatabase.from_uri(db_uri)
# llm = AzureOpenAI(deployment_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'), model_name=os.environ.get('OPENAI_DEPLOYMENT_NAME'), temperature=0)
 
# # Create SQL Database toolkit and LangChain Agent Executor
# execute_query = QuerySQLDataBaseTool(db=sql_db, llm=llm)
# agent_executor = create_sql_query_chain(llm, sql_db)
 
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
 
# # Define endpoints
# @app.get("/")
# async def homepage():
#     return "Welcome to SQL Query with LangChain"

# @app.post("/login")
# async def login(user: User, request: Request):
#     try:
#         # Find user in database
#         user_in_db = users.find_one({"username": user.username})
        
#         # If user doesn't exist, create new user
#         if not user_in_db:
#             new_user = {"username": user.username, "password": user.password}
#             inserted_user = users.insert_one(new_user)
#             user_id = str(inserted_user.inserted_id)
#         else:
#             # Verify password (in production, use proper password hashing)
#             if user.password != user_in_db.get("password"):
#                 raise HTTPException(status_code=400, detail="Incorrect password")
#             user_id = str(user_in_db["_id"])
        
#         # Generate token
#         expire = datetime.utcnow() + timedelta(minutes=15)
#         token_data = {"id": user_id, "username": user.username, "exp": expire}
#         token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
#         # Store user ID in session
#         request.session["user_id"] = user_id
        
#         return {"message": "Login successful", "token": token}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         user_id = payload.get("id")
#     except jwt.PyJWTError:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Could not validate credentials",
#         )
#     if user_id is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return user_id
 
# # Define a new endpoint to handle table selection
# @app.get("/tables")
# async def get_tables(user_id: int = Depends(get_current_user)):
#     if user_id is None:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     tables = sql_db.get_usable_table_names()
#     return {"tables": tables}
 
# # Define the endpoint to handle queries
# @app.post("/query")
# async def run_query(request: Request, user_id: int = Depends(get_current_user)):
#     if user_id is None:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     try:
#         data = await request.json()
#         question = data.get('question')
#         selected_tables = data.get('selected_tables', [])  # Get selected tables from request data
 
#         # Add selected tables to the question prompt
#         if selected_tables:
#             selected_tables_str = ", ".join([f"`{table}`" for table in selected_tables])
#             question += f" strictly using {selected_tables_str} tables"
 
#         # Pass user input to LangChain
#         query = agent_executor.invoke({"question": question})
 
#         # Execute the SQL query
#         result = execute_query.invoke({"query": query})
 
#         insights = llm.invoke(f"Analyze the following data and provide insights related to sales trends and projections. Do not generate content related to programming concepts or other irrelevant topics. The question asked was: {question}. The data fetched after executing the query is: {result}")
#         column_name = llm.invoke(f"Given this SQL query: {query}, and result as {result}.  Provide only the names of columns (in format ColumnName) as a Python list.have the column names in the same sequesnce as the result is being displayed , the result is a list of tuple , check for the first tuple about how much data is there , the no of column should not increase that count . Your response should only contain column names and nothing else, in the format ['column1', 'column2', 'column3', ...], without any additional explanations or prompts.")
#         insights = insights.replace("<|im_end|>","")
#         start_index = column_name.find('[')
#         end_index = column_name.find(']')
 
#         # Extract the content within the square brackets
#         extracted_content = column_name[start_index:end_index+1]  
 
#         # Convert string representation of list to actual list
#         if result and isinstance(result, str):
#             result = ast.literal_eval(result)
 
#         extracted_content_list = eval(extracted_content)
#         combined_result = [tuple(extracted_content_list)] + result
 
#         # Round up numeric values to two decimal places
#         for i in range(len(combined_result)):
#             if isinstance(combined_result[i], tuple):
#                 combined_result[i] = tuple(round(val, 2) if isinstance(val, float) else val for val in combined_result[i])      
 
#         combined_result_str = str(combined_result)
#         sql_history.insert_one({"id": user_id, "question": question, "query": query, "result": combined_result_str, "insights": insights})
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=400, detail=str(e))                    
 
#     return {"question": question, "query": query, "result":  combined_result, "insights": insights}
 
# # History of user
# @app.get("/history")
# async def get_history(user_id: int = Depends(get_current_user)):
#     try:
#         history_records = sql_history.find({"id": user_id})
#         history = {str(record["_id"]): dict(id=record["id"], question=record["question"], query=record["query"], result=record["result"], insights=record["insights"]) for record in history_records}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e)) 
 
#     return {"history": history}



from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase  # Updated import
from langchain_community.cache import SQLiteCache  # Updated import
from langchain.globals import set_llm_cache
import os
import ast
from dotenv import load_dotenv
from typing import List
import secrets
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from pymongo import MongoClient
import pathlib

load_dotenv()

app = FastAPI()

class User(BaseModel):
    username: str
    password: str  

origins = [
    "http://localhost:5173",
    "*"  # Allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)

client = MongoClient("mongodb://localhost:27017/")
db = client["user_db"]
users = db["users"]
sql_history = db["sql_history"]

# Database setup
data_dir = pathlib.Path("data")
data_dir.mkdir(exist_ok=True)
db_path = data_dir / "Chinook.db"
if not db_path.exists():
    db_path.touch()
    print(f"Created empty database file at {db_path}")

db_uri = f"sqlite:///{db_path.absolute()}"
sql_db = SQLDatabase.from_uri(db_uri)

# **Updated Azure OpenAI LLM Integration**
llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    model_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
# Create SQL Database toolkit and LangChain Agent Executor
execute_query = QuerySQLDataBaseTool(db=sql_db, llm=llm)
agent_executor = create_sql_query_chain(llm, sql_db)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.get("/")
async def homepage():
    return "Welcome to SQL Query with LangChain"

@app.post("/login")
async def login(user: User, request: Request):
    try:
        user_in_db = users.find_one({"username": user.username})
        if not user_in_db:
            new_user = {"username": user.username, "password": user.password}
            inserted_user = users.insert_one(new_user)
            user_id = str(inserted_user.inserted_id)
        else:
            if user.password != user_in_db.get("password"):
                raise HTTPException(status_code=400, detail="Incorrect password")
            user_id = str(user_in_db["_id"])

        expire = datetime.utcnow() + timedelta(minutes=15)
        token_data = {"id": user_id, "username": user.username, "exp": expire}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        request.session["user_id"] = user_id
        return {"message": "Login successful", "token": token}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("id")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_id

@app.get("/tables")
async def get_tables(user_id: int = Depends(get_current_user)):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    tables = sql_db.get_usable_table_names()
    return {"tables": tables}

@app.post("/query")
async def run_query(request: Request, user_id: int = Depends(get_current_user)):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        data = await request.json()
        question = data.get('question')
        selected_tables = data.get('selected_tables', []) 

        if selected_tables:
            selected_tables_str = ", ".join([f"`{table}`" for table in selected_tables])
            question += f" strictly using {selected_tables_str} tables"

        # **Generate SQL Query with LangChain**
        query = agent_executor.invoke({"question": question})

        # **Execute Query and Fetch Results**
        result = execute_query.invoke({"query": query})

        # **Generate Insights using LLM**
        insights_prompt = f"Analyze the following data and provide insights related to sales trends and projections. The question asked was: {question}. The data fetched is: {result}"
        insights = str(llm.invoke(insights_prompt))

        # **Extract Column Names**
        column_extraction_prompt = f"Given this SQL query: {query} and result: {result}, provide column names in a Python list format, strictly following the order of the result tuples."
        column_names = llm.invoke(column_extraction_prompt)

        # Extract and convert column names from string to list
        # start_index = column_names.find('[')
        # end_index = column_names.find(']')
        # extracted_content = column_names[start_index:end_index+1]  
        # extracted_content_list = eval(extracted_content)

        # # **Combine Column Names with Result**
        # if result and isinstance(result, str):
        #     result = ast.literal_eval(result)

        # combined_result = [tuple(extracted_content_list)] + result

        # # **Round numeric values**
        # for i in range(len(combined_result)):
        #     if isinstance(combined_result[i], tuple):
        #         combined_result[i] = tuple(round(val, 2) if isinstance(val, float) else val for val in combined_result[i])

        # **Store History**
        sql_history.insert_one({
            "id": user_id,
            "question": question,
            "query": query,
            "result": str(result),
            "insights": str(insights)
        })
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

    return {"question": question, "query": query, "result": result, "insights": insights}

@app.get("/history")
async def get_history(user_id: int = Depends(get_current_user)):
    try:
        history_records = sql_history.find({"id": user_id})
        history = {
            str(record["_id"]): {
                "id": record["id"],
                "question": record["question"],
                "query": record["query"],
                "result": record["result"],
                "insights": record["insights"]
            } for record in history_records
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 

    return {"history": history}
