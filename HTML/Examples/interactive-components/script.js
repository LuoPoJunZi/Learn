const tabs = document.querySelectorAll(".tab");
const tabContents = document.querySelectorAll(".tab-content");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    tabContents.forEach((item) => item.classList.remove("active"));

    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});

const faqQuestions = document.querySelectorAll(".faq-question");

faqQuestions.forEach((question) => {
  question.addEventListener("click", () => {
    const answer = question.nextElementSibling;
    answer.classList.toggle("open");
  });
});

const steps = document.querySelectorAll(".step");
const nextStepButton = document.getElementById("next-step");
let currentStep = 0;

nextStepButton.addEventListener("click", () => {
  currentStep = (currentStep + 1) % steps.length;
  steps.forEach((step, index) => {
    step.classList.toggle("active", index <= currentStep);
  });
});

const toast = document.getElementById("toast");
const showToastButton = document.getElementById("show-toast");

showToastButton.addEventListener("click", () => {
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 1800);
});

