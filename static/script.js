document.getElementById('feedbackForm').addEventListener('submit', function(event) {
    event.preventDefault();
    // Here you can add code to handle form submission, like AJAX request, etc.
    alert('Feedback submitted successfully!');
    // Optionally, you can redirect the user to another page after submission
    window.location.href = 'dash.html';
});
