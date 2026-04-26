from fastapi import Request


def get_container(request: Request):
    return request.app.state.container


def get_chat_service(request: Request):
    return request.app.state.container.chat_service


def get_kb_service(request: Request):
    return request.app.state.container.kb_service


def get_vector_store(request: Request):
    return request.app.state.container.vector_store