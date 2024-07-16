document.getElementById('terminal-button').addEventListener('click', function() {
    var elementToShow = document.getElementById('element-to-show');
    if (elementToShow.style.display === "none") {
        elementToShow.style.display = "block";
    } else {
        elementToShow.style.display = "none";
    }
});

document.getElementById('send-button').addEventListener('click', function() {
    var chatInput = document.getElementById('chat-input');
    var chatPanel = document.getElementById('chat-panel');

    // Create a new paragraph element for the message
    var message = document.createElement('p');
    message.textContent = chatInput.value;

    // Add the message to the chat panel
    chatPanel.appendChild(message);

    // Clear the chat input field
    chatInput.value = '';

    // Scroll to the bottom of the chat panel
    chatPanel.scrollTop = chatPanel.scrollHeight;
});
