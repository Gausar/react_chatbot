import { useRef } from "react";

const ChatForm = ({chatHistory, setChatHistory, generateBotResponse }) => {
    const inputRef = useRef();

    const handleFormSubmit = (e) => {
        e.preventDefault();
        const userMessage = inputRef.current.value.trim();
        if(!userMessage) return;
        inputRef.current.value = "";
        
        setChatHistory((history) => [...history, { role: "user", text: userMessage}]);
        // console.log(userMessage);

        
        setTimeout(() => {
            setChatHistory((history) => [...history, { role: "model", text: "Thinking..."}]);
            generateBotResponse([...chatHistory, {role: "model", text: `Using the details provided above, please address this query:  ${userMessage}. Give your answer in Mongolian!`}]);
            }, 600);
    }

    return (
        <form action="#" className="chat-form" onSubmit={handleFormSubmit}>
            <input ref={inputRef} type="text" placeholder="Message..." className="message-input" required />
            <button className="material-symbols-outlined">
            arrow_upward
            </button>
          </form>
    )
}

export default ChatForm;