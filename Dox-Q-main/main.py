from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from embedder import generate_vector_embeddings
from retriever import get_context_string
from llm import ask_llm
from database import (
    create_user,
    authenticate_user,
    get_user_by_id,
    get_user_documents,
    delete_document,
    save_chat,
    get_chat_history,
)
import os
import json
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
UPLOAD_DIRECTORY = "uploaded_pdfs"

# --- Basic Setup ---
app = FastAPI(title="doXQ Healthcare RAG", version="2.0.0")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "doxq-default-secret"),
    max_age=86400,  # 24 hours
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory and Template Setup ---
templates = Jinja2Templates(directory="templates")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

#  AUTH HELPERS

def get_current_user(request: Request) -> dict | None:
    """Extract user from session. Returns None if not logged in."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def require_auth(request: Request) -> dict:
    """Raise 401 if not authenticated."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

#  PAGE ROUTES (Frontend Serving)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard if logged in, otherwise to login."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def serve_login_page(request: Request):
    """Serves the login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def serve_register_page(request: Request):
    """Serves the registration page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_page(request: Request):
    """Serves the document management + upload page. Requires auth."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_page(request: Request):
    """Serves the chatbot interface page. Requires auth."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("chat.html", {"request": request})



#  AUTH API ENDPOINTS


@app.post("/api/register")
async def api_register(request: Request):
    """Create a new user account."""
    data = await request.json()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="All fields are required.")

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        user_id = create_user(username, email, password)
        return JSONResponse(content={"message": "Account created successfully", "user_id": user_id})
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/login")
async def api_login(request: Request):
    """Authenticate and create a session."""
    data = await request.json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    request.session["user_id"] = user["id"]
    return JSONResponse(content={"message": "Login successful", "user": user})


@app.post("/api/logout")
async def api_logout(request: Request):
    """Clear the session."""
    request.session.clear()
    return JSONResponse(content={"message": "Logged out successfully"})


@app.get("/api/me")
async def api_me(request: Request):
    """Get current authenticated user info."""
    user = require_auth(request)
    if user and user.get("created_at"):
        user["created_at"] = user["created_at"].isoformat()
    return JSONResponse(content={"user": user})



#  DOCUMENT API ENDPOINTS


@app.get("/api/documents")
async def api_list_documents(request: Request):
    """List all documents for the current user."""
    user = require_auth(request)
    docs = get_user_documents(user["id"])
    # Convert datetime objects to strings for JSON
    for doc in docs:
        if doc.get("upload_date"):
            doc["upload_date"] = doc["upload_date"].isoformat()
    return JSONResponse(content={"documents": docs})


@app.post("/api/upload-pdf")
async def api_upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Handles PDF upload: saves the file, then generates and stores
    its vector embeddings in MongoDB using SBERT.
    """
    user = require_auth(request)

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)

    print(f"Saving file to: {file_path}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"Triggering embedding pipeline for: {filename}")
        embedding_result = generate_vector_embeddings(filename, user_id=user["id"])

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded and embedded {filename}",
                "doc_id": embedding_result.get("doc_id"),
                "num_chunks": embedding_result.get("num_chunks"),
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def api_delete_document(request: Request, doc_id: int):
    """Delete a document and all its embeddings."""
    user = require_auth(request)
    deleted = delete_document(doc_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return JSONResponse(content={"message": "Document deleted successfully"})



#  CHAT API ENDPOINTS


@app.post("/api/chat")
async def api_chat(request: Request):
    """
    Full RAG pipeline: retrieves relevant chunks from MongoDB
    (only for selected documents), then passes them + the user
    question to Ollama for an answer.
    """
    user = require_auth(request)
    data = await request.json()
    user_message = data.get("message", "")
    document_ids = data.get("document_ids", [])

    print(f"Received message: {user_message}")
    print(f"Querying documents: {document_ids}")

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if not document_ids:
        raise HTTPException(status_code=400, detail="No documents selected for querying.")

    try:
        # 1. Retrieve relevant chunks from MongoDB (only selected docs)
        context = get_context_string(user_message, document_ids=document_ids, k=10)

        if not context:
            bot_reply = "I couldn't find relevant information in the selected documents. Please try a different question or select different documents."
        else:
            # 2. Pass context + question to Ollama
            bot_reply = ask_llm(user_question=user_message, context=context)

        # 3. Save to chat history
        save_chat(user["id"], user_message, bot_reply, document_ids)

    except Exception as e:
        import traceback
        traceback.print_exc()
        bot_reply = f"Error: {str(e)}"

    print(f"Bot reply: {bot_reply}")
    return JSONResponse(content={"reply": bot_reply})


@app.get("/api/chat-history")
async def api_chat_history(request: Request):
    """Get recent chat history for the current user."""
    user = require_auth(request)
    history = get_chat_history(user["id"])
    # Convert datetime objects
    for item in history:
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        if item.get("document_ids") and isinstance(item["document_ids"], str):
            item["document_ids"] = json.loads(item["document_ids"])
    return JSONResponse(content={"history": history})
