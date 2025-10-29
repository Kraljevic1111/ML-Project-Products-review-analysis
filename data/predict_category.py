import joblib
import pandas as pd

model = joblib.load("data/model/product classifier model.pkl")
print("Model succesfully loaded")
print("Type exit to exit the program in any moment")

while True:
    title = input("Enter a product title:")
    if title == "exit":
        print("Exiting...")
        break
    user_input = pd.DataFrame([{"Product Title": title}])
    
    prediction = model.predict(user_input)[0]
    print(f"Predicted category:{prediction}")