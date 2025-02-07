document.addEventListener("DOMContentLoaded", function () {
    
    const tabs = document.querySelectorAll(".tab");
    const contents = document.querySelectorAll(".content");

    tabs.forEach(tab => {
        tab.addEventListener("click", function () {
           
            tabs.forEach(t => t.classList.remove("active"));
            
            this.classList.add("active");

            contents.forEach(content => content.classList.remove("active"));

            const targetId = this.getAttribute("data-target");
            document.getElementById(targetId).classList.add("active");
        });
    });
});
