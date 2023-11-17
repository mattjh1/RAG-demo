import chainlit as cl
from model import qa_bot
from loguru import logger


@cl.on_chat_start
async def start():
    logger.debug("Into decorator async def start")
    chain = qa_bot()  
    msg = cl.Message(content="Bot is initiating...") 
    await msg.send()
    msg.content = "Hello! How can I help?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    logger.debug("Into decorator async def main")
    chain = cl.user_session.get("chain")
    logger.info(f"Chain :: {chain} ")
    # Callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]  # Storing source doc page

    if sources:
        for source_doc in sources:
            metadata = source_doc.metadata if hasattr(source_doc, "metadata") else {}
            source = metadata.get("source", None)
            page = metadata.get("page", None)
            if page is None or source is None:
                continue

            answer += f"\n\nSource: {source}\nPage: {page}\n"
    else:
        answer += "\nNo sources found"

    logger.info(f"sources :: {sources}")
    logger.info(f"async def main, res:: {answer} ")
    await cl.Message(content=answer).send()
