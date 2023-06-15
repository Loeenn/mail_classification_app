  let circularProgress = document.querySelector(".circular-progress"),
    progressValue = document.querySelector(".progress-value");

let progressStartValue = 0,    
    progressEndValue = 100,
    speed = 100;
    
let progress = setInterval(() => {
    progressStartValue++;

    progressValue.textContent = `${progressStartValue}%`
    circularProgress.style.background = `conic-gradient(#3285ea ${progressStartValue * 3.6}deg, #ededed 0deg)`

    if(progressStartValue == progressEndValue){
            var h1 = document.createElement('h1');
    h1.appendChild(document.createTextNode("Обучение завершено. Посмотрите обновлённые результаты"));
    document.body.appendChild(h1);
        clearInterval(progress);
    }    
}, speed);
 