## 사용방법: 
###
POST http://localhost:3000/set_user_info
Content-Type: application/json

{
  "age": 70, 
  "like": "운동" 

}
###
POST http://localhost:3000/chat
Content-Type: application/json

{
  "text": "나의 관심사가 뭐야", 
  "user": "0000"
}

- age: number
- like: string
- text: string
- user: string

