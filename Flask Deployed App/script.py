import speech_recognition as sr
import pandas as pd
import pyttsx3

# Initialize the speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Load the plant disease dataset
disease_data = pd.read_csv('disease_info.csv')

# Print the column names to verify
print("Columns in the dataset:", disease_data.columns)

# Strip leading/trailing spaces and convert to lowercase for comparison
disease_data['disease_name'] = disease_data['disease_name'].str.strip().str.lower()

# Print the disease names for debugging
print("Disease names in the dataset:")
print(disease_data['disease_name'])

def normalize_text(text):
    """Normalize text by stripping leading/trailing spaces and converting to lowercase."""
    return text.strip().lower()

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_disease_details(disease_name):
    """Fetch disease details from the dataset based on disease name."""
    disease_name_normalized = normalize_text(disease_name)
    disease = disease_data[disease_data['disease_name'] == disease_name_normalized]
    if disease.empty:
        return None
    return disease.iloc[0]

def main():
    with sr.Microphone() as source:
        print("Please speak the name of the plant disease you want to know about:")
        audio = recognizer.listen(source)

        try:
            # Convert speech to text
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")

            # Normalize query
            query_normalized = normalize_text(query)

            # Fetch disease details
            disease_details = get_disease_details(query_normalized)
            if disease_details is not None:
                details = (
                    f"Disease: {disease_details['disease_name']}\n"
                    f"Description: {disease_details['description']}\n"
                    f"Possible Steps: {disease_details['Possible Steps']}\n"
                    f"Image URL: {disease_details['image_url']}"
                )
                print(details)
                speak(details)
            else:
                response = "Sorry, I couldn't find any information about that disease."
                print(response)
                speak(response)
        except sr.UnknownValueError:
            response = "Sorry, I could not understand your speech."
            print(response)
            speak(response)
        except sr.RequestError as e:
            response = f"Could not request results; {e}"
            print(response)
            speak(response)

if _name_ == "_main_":
    main()