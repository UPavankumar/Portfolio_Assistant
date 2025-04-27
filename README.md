# Portfolio Assistant

A Streamlit-based chatbot assistant for my portfolio, powered by Alfred Pennyworth's wit and charm.

## Features

* **Portfolio Insights**: Responds to user queries about my portfolio, providing information on my skills, experience, and projects.
* **Conversational Interface**: Engages users with a conversational interface, making it easy to explore my portfolio.
* **Streamlit Integration**: Built with Streamlit, allowing for seamless deployment and sharing.

## Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **Deployment**: Streamlit Cloud
* **Integration**: Firebase, Groq API

## Requirements

* Python 3.8+
* See `requirements.txt` for dependencies

## Getting Started

1. Clone the repository: `git clone https://github.com/UPavankumar/Portfolio_Assistant.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
	* Create a `.env` file with the following variables:
		+ `GROQ_API_KEY=YOUR_GROQ_API_KEY`
		+ `FIREBASE_PROJECT_ID=YOUR_FIREBASE_PROJECT_ID`
		+ `FIREBASE_PRIVATE_KEY=YOUR_FIREBASE_PRIVATE_KEY`
4. Set up Firebase:
	* Create a Firebase project
	* Enable Firestore database
	* Set up Firebase Admin SDK
5. Run the app locally: `streamlit run app.py`

## Deployment

Deployed on Streamlit Cloud, integrated with Firebase portfolio and Groq API.

## How it Works

1. User interacts with the chatbot interface
2. Chatbot responds with relevant information from the portfolio using Groq API
3. Data is retrieved from Firestore database

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you'd like to change.
