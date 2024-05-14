
import google.generativeai as genai

GOOGLE_API_KEY="AIzaSyDO3yKGP_m1bhXwBFJVeJrgdDmVigVDu98"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Will Sin make the academic come back?")
print(response.text)