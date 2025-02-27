import json

def save_user_data(user_data):
    """
    사용자 정보를 저장하는 서비스 로직
    """
    try:
        with open("user_data.json", "a") as file:
            file.write(json.dumps(user_data.dict()) + "\n")
        print("User data saved successfully.")
    except Exception as e:
        print(f"Error saving user data: {e}")
