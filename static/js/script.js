document.addEventListener('DOMContentLoaded', function() {
    const fileUpload = document.getElementById('file-upload');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatTextArea = document.getElementById('chat-text-area');

    const modelSelect = document.getElementById('model-select');
    const modelDropdown = document.getElementById('model-dropdown');
    const selectedModel = modelSelect.querySelector('.selected-model');
    
    const noChatdisplay = document.querySelector('.no-chat-display');

    let pdfUploaded = false;

    selectedModel.addEventListener('click', function(event) {
        event.stopPropagation();
        modelDropdown.style.display = modelDropdown.style.display === 'flex' ? 'none' : 'flex';
        rotateDropdownIcon();
    });
    
    document.addEventListener('click', function(event) {
        if (!modelSelect.contains(event.target)) {
            modelDropdown.style.display = 'none';
            rotateDropdownIcon(false);
        }
    });

    modelDropdown.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', function(event) {
            event.stopPropagation();
            selectedModel.firstChild.textContent = this.textContent;
            currentModel = this.dataset.value;
            modelDropdown.style.display = 'none';
            rotateDropdownIcon(false);
            
            // Send selected model to Flask backend
            fetch('/update_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model: currentModel })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    addSystemMessage(`Model switched to ${this.textContent}`);
                } else {
                    addSystemMessage('Failed to switch model');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addSystemMessage('Failed to switch model');
            });
        });
    });


    function rotateDropdownIcon(open = true) {
        const icon = selectedModel.querySelector('svg');
        icon.style.transform = open ? 'rotate(90deg)' : 'rotate(0deg)';
        icon.style.transition = 'transform 0.3s ease';
    }

    fileUpload.addEventListener('change', async function(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload_pdf', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    pdfUploaded = true;
                    addSystemMessage('PDF uploaded successfully. You can now ask questions about its content.');
                    noChatdisplay.style.display = 'none';
                } else {
                    addSystemMessage('Failed to upload PDF. Please try again.');
                }
            } catch (error) {
                console.error('Error:', error);
                addSystemMessage('An error occurred while uploading the PDF.');
            }
        }
    });


    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    async function sendMessage() {
        if (!pdfUploaded) {
            addSystemMessage('Please upload a PDF first.');
            return;
        }
    
        const question = chatInput.value.trim();
        if (question) {
            addUserMessage(question);
            chatInput.value = '';
    
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `question=${encodeURIComponent(question)}`
                });
    
                if (response.ok) {
                    const answer = await response.text();
                    addAIMessage(answer);
                } else {
                    addSystemMessage('Failed to get a response. Please try again.');
                }
            } catch (error) {
                console.error('Error:', error);
                addSystemMessage('An error occurred while processing your question.');
            }
        }
    }

    function addUserMessage(message) {
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'human';
        userMessageDiv.textContent = message;
        chatTextArea.appendChild(userMessageDiv);
        chatTextArea.scrollTop = chatTextArea.scrollHeight;
    }

    function addAIMessage(message) {
        const aiMessageDiv = document.createElement('div');
        aiMessageDiv.className = 'ai';
        aiMessageDiv.textContent = message;
        chatTextArea.appendChild(aiMessageDiv);
        chatTextArea.scrollTop = chatTextArea.scrollHeight;
    }

    function addSystemMessage(message) {
        const systemMessageDiv = document.createElement('div');
        systemMessageDiv.className = 'system';
        systemMessageDiv.textContent = message;
        chatTextArea.appendChild(systemMessageDiv);
        chatTextArea.scrollTop = chatTextArea.scrollHeight;
    }

});