import os
import json
# from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr # type: ignore

import base64
from io import BytesIO
from PIL import Image


from pydub import AudioSegment
from pydub.playback import play

# Initialization

# load_dotenv(override=True)

openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

ticket_prices = {
        "london": {
            "price":"$499",
            "currency":"USD",
            "number_of_avaiable_tickets":5,
        },
        "paris": {
            "price":"$399",
            "currency":"USD",
            "number_of_avaiable_tickets":2,
        },
        "delhi": {
            "price":"2599",
            "currency":"INR",
            "number_of_avaiable_tickets":0,
        },
        "tokyo": {
            "price":"$699",
            "currency":"USD",
            "number_of_avaiable_tickets":3,
        },
        "sydney": {
            "price":"$799",
            "currency":"USD",
            "number_of_avaiable_tickets":1,
        }
                 
    }

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    if city not in ticket_prices:
        return "Unknown"
    if ticket_prices[city]['number_of_avaiable_tickets'] == 0:
        return "Sold out"   
    return ticket_prices.get(city)['price']

def book_ticket(destination_city):
    print(f"Tool book_ticket called for {destination_city}")
    city = destination_city.lower()
    if city not in ticket_prices:
        return "Unknown"
    if ticket_prices[city]['number_of_avaiable_tickets'] == 0:
        return "Sold out"   
    ticket_prices[city]['number_of_avaiable_tickets'] -= 1
    return "Ticket booked successfully!"

# print(get_ticket_price("London"))


price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

book_ticket_function = {
    "name": "book_ticket",
    "description": "book the ticket for destination city. if no tickets available then give sold out. Call this whenever you need to book the ticket, for example when a customer asks 'can you please book the ticket for me'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": book_ticket_function},
    ]


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    if function_name == "book_ticket":
        response = book_ticket(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city,"response": response}),
            "tool_call_id": tool_call.id
        }
        return response, city
    elif function_name == "get_ticket_price":
        # Get the ticket price
        print(f"Tool get_ticket_price called for {city}")
        response = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city,"response": response}),
            "tool_call_id": tool_call.id
        }
        return response, city



def artist(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="onyx",    # Also, try replacing onyx with alloy
      input=message
    )
    
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)
    
# talker("Well, hi there")


def transcribe_and_ask(audio_file_path,history):
    try:
        # Check if audio_file_path is None
        if audio_file_path is None:
            return "No audio provided", "Please record some audio using the microphone."
            
        audio_file = open(audio_file_path, "rb")
        response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)        
        transcript = response.text
        print(f"response: {response}")
        print(f"transcript: {transcript}")
        question = transcript
        
        # Step 2: Ask the question to GPT
        print(f"Question: {question}")
        
        answer, artist_image = chat(question,history)
        
        return question, answer, artist_image
    except Exception as e:
        error_message = f"An error occurred: {e}"
        return error_message, error_message  # Return two values to match the interface expectations


# interface = gr.Interface(
#     fn=transcribe_and_ask,
#     inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak your question"),  # Using 'type="file"' here
#     outputs=[
#         gr.Textbox(label="You asked"),
#         gr.Textbox(label="GPT's answer")
#     ],
#     title="Voice to GPT Chat"
# )


def chat(message, history):
    artist_image = None
    # Initialize history as empty list if None
    if history is None:
        history = []
    
    # If message is a list (chatbot history format), extract last user message
    if isinstance(message, list) and message:
        last_message = message[-1]["content"] if isinstance(message[-1], dict) and "content" in message[-1] else ""
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": last_message}]
    else:
        # Handle direct message string
        messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        artist_image = artist(city)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    # Comment out or delete the next line if you'd rather skip Audio for now..
    talker(reply)
    
    return reply, artist_image


with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    #  Add audio input component
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Or speak your question")
    
    with gr.Row():
        clear = gr.Button("Clear")
        speak_button = gr.Button("Process Speech")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history
    
    def process_speech(audio, history):
        
        question, answer, artist_image = transcribe_and_ask(audio,history)
        
        # Update history with the transcribed question and AI's answer
        new_history = history.copy() if history else []
        
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "assistant", "content": answer})
        print(f"History: {new_history}")
        return new_history, artist_image
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
         chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    speak_button.click(
        process_speech, 
        inputs=[audio_input, chatbot], 
        outputs=[chatbot, image_output]
    )
    
    # Clear button functionality
    clear.click(lambda: ([], None), outputs=[chatbot, image_output])

ui.launch(inbrowser=True)
