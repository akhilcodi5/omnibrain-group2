from graph import graph


user_input = input("Enter your input: ")

result = graph.invoke({
    "user_input": user_input,
    "response": ""
})

print("Output:", result["response"])