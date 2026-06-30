from llm.rag import ask

question = input("Ask a question: ")
response = ask(question)

print()
print(response["answer"])
print()

print("Sources:")
for source in response["sources"]:
    print(f'{source["filename"]} (Page {source["page"]})')
