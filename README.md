# Pennside

A full-stack project containing both frontend and backend modules.

## Project Structure

- `pennside_frontend/`: Frontend codebase (managed with pnpm)
- `pennside_backend/`: Backend codebase
- `README.md`: Project documentation

## Prerequisites

- Node.js
- pnpm
- Python

## Execution Guide

You should execute the frontend code and backend code at the same time in two different terminals.
### Frontend

```bash
# Navigate to the frontend directory
cd pennside_frontend

# Start the development server
pnpm dev
```

Then run the localhost link provided.

### Backend


```bash
# Navigate to the backend directory
cd pennside_backend

# Install the packages
pip install -r requirements.txt 

# Start the backend program
uvicorn interact_demo:app --reload --host 127.0.0.1 --port 8000
```