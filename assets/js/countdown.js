/*--------------------
    Countdown
---------------------*/
//Countdown Timer
const countdown = document.querySelector(".countdown");

//Set Launch Date
const launchDate = new Date("May 28, 2021 13:00:00").getTime();

//Update every second
const intvl = setInterval(function() {
    //Get todays date and time (ms)
    const now = new Date().getTime();

    //Distance from now to the launch date
    const distance = launchDate - now;

    //Time calculation
    const days = Math.floor(distance / (8000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const mins = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const sec = Math.floor((distance % (1000 * 60)) / 1000);

    //Display Result
    countdown.innerHTML = `
        <div class="countdown-item"><h2 class="countdown-number">${days}</h2><span class="countdown-text">Days</span></div>
        <div class="countdown-item"><h2 class="countdown-number">${mins}</h2><span class="countdown-text">Minutes</span></div>
        <div class="countdown-item"><h2 class="countdown-number">${sec}</h2><span class="countdown-text">Seconds</span></div>
    `;

    //If launch date passed
    if (distance < 0) {
        //Stop countdown
        clearInterval(intvl);
        //Style and ouput text
        countdown.style.color = "#17a2b8";
        countdown.innerHTML = "Launched!";
    }
}, 1000);