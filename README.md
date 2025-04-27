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
* **Integration**: Firebase

## Requirements

* Python 3.8+
* Streamlit 1.20+
* Firebase Admin SDK
* Other dependencies:
	+ `streamlit-chat`
	+ `google-cloud-firestore`

## Getting Started

1. Clone the repository: `git clone https://github.com/UPavankumar/Portfolio_Assistant.git`
2. Install dependencies:
	* `pip install streamlit`
	* `pip install streamlit-chat`
	* `pip install google-cloud-firestore`
	* `pip install firebase-admin`
3. Set up Firebase:
	* Create a Firebase project
	* Enable Firestore database
	* Set up Firebase Admin SDK
4. Run the app locally: `streamlit run app.py`

## Deployment

Deployed on Streamlit Cloud, integrated with Firebase portfolio.

## How it Works

1. User interacts with the chatbot interface
2. Chatbot responds with relevant information from the portfolio
3. Data is retrieved from Firestore database

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you'd like to change.
