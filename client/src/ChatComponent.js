import React, { useState } from "react";
import "./styles.css";
//import axios from 'axios';

const ChatComponent = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async () => {
    // Sending the user message to the backend
    const userMessage = { text: input, type: "user" };
    setMessages([...messages, userMessage]);


    try {
      const response = await fetch(
        "http://localhost:3000/chatbot",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message: input }),
        }
      );

      // Assuming the response from the backend is in JSON format
      const responseData = await response.json();

      const botMessage = { text: responseData.reply, type: "bot" };
      // setMessages(messages=> [...messages, { text: input, type: "user" }, botMessage]);
      setMessages(messages=> [...messages, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
    }

    setInput("");
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${message.type === "user" ? "user" : "bot"
              }`}
          >
            {message.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          placeholder="Type a message..."
          value={input}
          onChange={handleInputChange}
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>
    </div>
  );
};

export default ChatComponent;

// import React, { useState } from 'react';
// import './styles.css';  // Import the CSS file

// const ChatComponent = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const handleInputChange = (e) => {
//     setInput(e.target.value);
//   };

//   const handleSendMessage = () => {
//     if (input.trim() === '') return;

//     setMessages([...messages, { text: input, type: 'user' }]);
//     setInput('');
//   };

//   return (
//     <div className="chat-container">
//       <div className="chat-messages">
//         {messages.map((message, index) => (
//           <div key={index} className={message.type}>
//             {message.text}
//           </div>
//         ))}
//       </div>
//       <div className="chat-input">
//         <input
//           type="text"
//           placeholder="Type a message..."
//           value={input}
//           onChange={handleInputChange}
//         />
//         <button onClick={handleSendMessage}>Send</button>
//       </div>
//     </div>
//   );
// };

// export default ChatComponent;
