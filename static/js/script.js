document.addEventListener('DOMContentLoaded', function() {
    // --- UI Elements ---
    const textModeBtn = document.getElementById('textMode');
    const voiceModeBtn = document.getElementById('voiceMode');
    const textInterface = document.getElementById('textInterface');
    const voiceInterface = document.getElementById('voiceInterface');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitQuery');
    const recordBtn = document.getElementById('recordButton');
    const recordingStatus = document.querySelector('.recording-status');
    const recordingStatusText = document.getElementById('recordingStatusText');
    const responseArea = document.getElementById('responseArea');
    const responseText = document.getElementById('responseText');
    const hindiResponse = document.getElementById('hindiResponse');
    const sourcesArea = document.getElementById('sourcesArea'); // Updated ID
    const sourcesList = document.getElementById('sourcesList');
    const responseAudio = document.getElementById('responseAudio');
    const regionSelect = document.getElementById('regionSelect'); // Region dropdown
    const loadingIndicator = document.getElementById('loadingIndicator'); // Loading indicator
    const transcriptionArea = document.getElementById('transcriptionArea'); // Area to show transcription
    const transcriptionText = document.getElementById('transcriptionText');
    const healthStatusDiv = document.getElementById('healthStatus');

    // --- State Variables ---
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // --- Event Listeners ---
    textModeBtn.addEventListener('click', () => switchMode('text'));
    voiceModeBtn.addEventListener('click', () => switchMode('voice'));
    submitBtn.addEventListener('click', submitTextQuery);
    queryInput.addEventListener('keypress', handleTextareaKeypress);
    recordBtn.addEventListener('click', toggleRecording);

    // --- Initialization ---
    checkHealth(); // Check system health on load
    switchMode('text'); // Default to text mode

    // --- Functions ---

    function switchMode(mode) {
        hideElement(responseArea);
        hideElement(loadingIndicator);
        hideElement(transcriptionArea);
        hideElement(responseAudio);
        stopRecording(); // Stop recording if switching modes

        if (mode === 'text') {
            textModeBtn.classList.add('active');
            voiceModeBtn.classList.remove('active');
            textInterface.classList.remove('hidden');
            voiceInterface.classList.add('hidden');
            queryInput.focus();
        } else { // voice mode
            voiceModeBtn.classList.add('active');
            textModeBtn.classList.remove('active');
            voiceInterface.classList.remove('hidden');
            textInterface.classList.add('hidden');
        }
    }

    function handleTextareaKeypress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent newline
            submitTextQuery();
        }
    }

    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
             alert('Microphone access is not supported by your browser.');
             return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream); // Use default mimeType or specify if needed
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                isRecording = false;
                updateRecordingUI(false); // Update UI immediately on stop

                if (audioChunks.length > 0) {
                    const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' }); // Use recorded mimeType
                    await submitVoiceQuery(audioBlob);
                } else {
                     console.warn("Recording stopped with no audio data.");
                     hideElement(loadingIndicator); // Hide loading if no data
                }
                 // Clean up stream tracks
                stream.getTracks().forEach(track => track.stop());
                mediaRecorder = null; // Release recorder instance
            };

             mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                alert(`Recording error: ${event.error.name}. Please try again.`);
                isRecording = false;
                updateRecordingUI(false);
                hideElement(loadingIndicator);
            };


            mediaRecorder.start();
            isRecording = true;
            updateRecordingUI(true);
            hideElement(responseArea); // Hide previous response
            hideElement(transcriptionArea);
            hideElement(responseAudio);


        } catch (error) {
            console.error('Error accessing microphone:', error);
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                alert('Microphone access denied. Please grant permission in your browser settings.');
            } else {
                alert('Could not access microphone. Please ensure it is connected and permissions are granted.');
            }
            isRecording = false;
             updateRecordingUI(false);
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            try {
                 mediaRecorder.stop(); // This triggers the onstop event
                 // UI update happens in onstop
                 showElement(loadingIndicator, 'flex'); // Show loading while processing starts
                 recordingStatusText.textContent = 'Processing...';

            } catch (error) {
                console.error("Error stopping recorder:", error);
                 // Manually update UI if stop fails somehow
                isRecording = false;
                updateRecordingUI(false);
                hideElement(loadingIndicator);
            }
        }
    }

     function updateRecordingUI(isRec) {
         recordBtn.textContent = isRec ? 'Stop Recording' : 'Record Question';
         if (isRec) {
             showElement(recordingStatus, 'flex'); // Use flex display
             recordingStatusText.textContent = 'Recording...';
         } else {
             hideElement(recordingStatus);
         }
         recordBtn.disabled = false; // Ensure button is enabled after state change
     }


    async function submitTextQuery() {
        const query = queryInput.value.trim();
        const region = regionSelect.value; // Get selected region
        if (!query) return;

        disableSubmitButton(true);
        showElement(loadingIndicator, 'flex'); // Use 'flex' for display
        hideElement(responseArea);

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json' // Indicate we expect JSON back
                },
                body: JSON.stringify({
                    query: query,
                    region: region, // Send selected region
                    user_id: 'web_user_' + Date.now() // Example user ID
                })
            });

            const data = await response.json();
            if (!response.ok) {
                 console.error("API Error:", data);
                 throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            displayResponse(data);
            queryInput.value = ''; // Clear input on success

        } catch (error) {
            console.error('Error submitting text query:', error);
            displayError(`Failed to get response: ${error.message}`);
        } finally {
            disableSubmitButton(false);
            hideElement(loadingIndicator);
        }
    }

    async function submitVoiceQuery(audioBlob) {
        const formData = new FormData();
        const region = regionSelect.value; // Get selected region
        formData.append('audio', audioBlob, `query_${Date.now()}.webm`); // Use timestamp in filename, adjust extension if needed
        formData.append('region', region); // Send region as form data
        formData.append('user_id', 'voice_user_' + Date.now());

        // Keep loading indicator shown from stopRecording()

        try {
            const response = await fetch('/api/speech-query', {
                method: 'POST',
                body: formData
                // No Content-Type header needed for FormData, browser sets it
            });

            const data = await response.json();
             if (!response.ok) {
                 console.error("API Error:", data);
                 throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            displayResponse(data);

            // Display transcription if available
            if (data.transcription) {
                transcriptionText.textContent = data.transcription;
                showElement(transcriptionArea);
            } else {
                 hideElement(transcriptionArea);
            }

            // Play audio response if URL provided
            if (data.audio_url) {
                responseAudio.src = data.audio_url;
                showElement(responseAudio);
                // Optional: Auto-play (might be blocked by browser)
                // responseAudio.play().catch(e => console.log("Autoplay prevented:", e));
            } else {
                 hideElement(responseAudio);
            }

        } catch (error) {
            console.error('Error submitting voice query:', error);
            displayError(`Failed to process audio: ${error.message}`);
             hideElement(transcriptionArea); // Hide transcription area on error
             hideElement(responseAudio);
        } finally {
             hideElement(loadingIndicator); // Hide loading indicator after processing
        }
    }

    function displayResponse(data) {
        showElement(responseArea); // Show the whole response area
        hideElement(responseAudio); // Hide audio initially

        // Clear previous content
        responseText.innerHTML = '';
        hindiResponse.innerHTML = '';
        sourcesList.innerHTML = '';
        hideElement(hindiResponse); // Hide Hindi section by default
        hideElement(sourcesArea); // Hide sources section by default


        if (!data || data.success === false) {
             // Handle explicit failure or errors passed from backend
             const errorMsg = data?.error || 'An unknown error occurred.';
             const answerEng = data?.answer || 'Processing failed.';
             const answerHin = data?.hindi_answer; // Use provided Hindi error or fallback

             responseText.innerHTML = `<p class="error">${answerEng} (Error: ${errorMsg})</p>`;
              if (answerHin) {
                  hindiResponse.innerHTML = `<p class="error">${answerHin}</p>`;
                  showElement(hindiResponse);
              }
            return; // Stop processing further
        }

        // --- Display successful response ---

        // English Answer
        if (data.answer) {
            // Sanitize slightly - replace newlines with <br> for display
            responseText.innerHTML = `<p>${data.answer.replace(/\n/g, '<br>')}</p>`;
        } else {
             responseText.innerHTML = `<p><i>No direct answer generated. Check sources if available.</i></p>`;
        }


        // Hindi Answer (only if provided by backend)
        if (data.hindi_answer) {
            hindiResponse.innerHTML = `<p>${data.hindi_answer.replace(/\n/g, '<br>')}</p>`;
            showElement(hindiResponse);
        }

        // Sources
        if (data.sources && data.sources.length > 0) {
            sourcesList.innerHTML = data.sources.map(source => {
                // Basic sanitization for display
                const safeContent = source.content ? escapeHtml(source.content) : 'No content preview available.';
                const sourceName = source.source ? escapeHtml(source.source) : 'Unknown Source';
                const pageInfo = source.page ? ` (Page: ${escapeHtml(source.page)})` : '';
                // const sectionInfo = source.section ? `<br>Section: ${escapeHtml(source.section)}` : ''; // If section exists

                return `<li>
                    <strong>${sourceName}${pageInfo}</strong>
                    <p class="excerpt">${safeContent}</p>
                </li>`;
            }).join('');
             showElement(sourcesArea);
        } else {
             // Optionally display "No sources" message or just hide the area
             // sourcesList.innerHTML = '<li>No relevant sources found.</li>';
             // showElement(sourcesArea);
             hideElement(sourcesArea); // Prefer hiding if no sources
        }

         // Handle audio URL (might come from voice or text if TTS added later)
         if (data.audio_url) {
            responseAudio.src = data.audio_url;
            showElement(responseAudio);
        }

    }

    function displayError(message = "An unexpected error occurred.") {
        showElement(responseArea);
        responseText.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
        hindiResponse.innerHTML = `<p class="error hindi-text">क्षमा करें, एक त्रुटि हुई।</p>`; // Generic Hindi error
        showElement(hindiResponse);
        sourcesList.innerHTML = ''; // Clear sources on error
        hideElement(sourcesArea);
        hideElement(responseAudio); // Hide audio player on error
    }


    // --- Utility Functions ---

     function disableSubmitButton(disabled) {
        submitBtn.disabled = disabled;
        submitBtn.textContent = disabled ? 'Processing...' : 'Submit';
     }

    function showElement(element, displayType = 'block') {
        if (element) {
            element.style.display = displayType;
            element.classList.remove('hidden');
        }
    }

    function hideElement(element) {
        if (element) {
            element.style.display = 'none';
            element.classList.add('hidden');
        }
    }

    function escapeHtml(unsafe) {
        if (!unsafe) return "";
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }
    

     async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            let healthy = true;
            let statusMsg = "System Status: ";

            // Check critical components based on health endpoint response
            if (data.components?.embeddings !== 'initialized') {
                 healthy = false; statusMsg += "Embeddings Error. "; }
            if (data.components?.query_engine !== 'initialized') {
                 healthy = false; statusMsg += "Query Engine Error. "; }
            if (data.components?.google_api_configured === false) {
                 healthy = false; statusMsg += "Google API Key Missing. "; }
             // Speech is optional for text mode
            if (data.components?.speech_recognition !== 'initialized') {
                 statusMsg += "Speech Recognition Offline. "; }


            if (healthy) {
                 healthStatusDiv.textContent = statusMsg + "OK";
                 healthStatusDiv.className = 'health-status healthy';
            } else {
                 healthStatusDiv.textContent = statusMsg + "Issues Detected.";
                 healthStatusDiv.className = 'health-status unhealthy';
            }

        } catch (error) {
             console.error("Health check failed:", error);
             healthStatusDiv.textContent = "System Status: Unknown (Health Check Failed)";
             healthStatusDiv.className = 'health-status unhealthy';
        }
    }

});


