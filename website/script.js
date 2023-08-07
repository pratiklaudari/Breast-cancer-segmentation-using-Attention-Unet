const slideshowContainer = document.querySelectorAll(".slideshow");
const slideshowImages = [
    "fig1.png",
    "fig3.png",
    // Add more image URLs as needed
];
let currentSlideIndex = 0;


function startHomeSlideshow() {
    setInterval(() => {
        currentSlideIndex = (currentSlideIndex + 1) % slideshowImages.length;
        const newImageSrc = slideshowImages[currentSlideIndex];
        const heroImage = slideshowContainer[0].querySelector("img");
        heroImage.src = newImageSrc;
    }, 3000); // Change the interval to control the time between slides (3000ms = 3 seconds)
}

startHomeSlideshow();
