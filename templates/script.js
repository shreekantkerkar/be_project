document.addEventListener("DOMContentLoaded", function() {
    const sections = document.querySelectorAll('section');
    const fileInput = document.querySelector('input[type="file"][name="resume"]');
    const resumeCheckbox = document.getElementById('resumeCheckbox');
    const descriptionCheckbox = document.getElementById('descriptionCheckbox');
    const startButton = document.querySelector('.button.start-now');

    const checkVisibility = () => {
        const triggerBottom = window.innerHeight * 0.8;

        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;

            if (sectionTop < triggerBottom) {
                section.classList.add('is-visible');
            } else {
                section.classList.remove('is-visible');
            }
        });
    };

    window.addEventListener('scroll', checkVisibility);
    checkVisibility();

    // Resume upload change event listener
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            resumeCheckbox.checked = true;
            checkButtonState();
        }
    });

    // Update word count and check description checkbox
    function updateWordCount() {
        var textEntered = document.getElementById('resumeText').value;
        var wordCount = textEntered.split(/\s+/).filter(function(n) { return n != '' }).length;
        if (wordCount > 500) {
            var truncatedText = textEntered.split(/\s+/).slice(0, 500).join(' ');
            document.getElementById('resumeText').value = truncatedText;
            wordCount = 500;
        }
        document.getElementById('wordCount').textContent = wordCount + '/500';
        descriptionCheckbox.checked = wordCount > 0;
        // No need to call checkButtonState here unless the button state depends on the job description being filled
    }

    document.getElementById('resumeText').addEventListener('input', updateWordCount);

    // Check if the start button should be enabled
    function checkButtonState() {
        // Button is enabled if resume is uploaded
        startButton.disabled = !resumeCheckbox.checked;
    }

    // Initial call to update word count and set initial button state
    updateWordCount();
    checkButtonState();

    const startInterviewButton = document.getElementById('startInterviewButton');

    startInterviewButton.addEventListener('click', function() {
        window.location.href = 'resumeparser.html';
    });
});
