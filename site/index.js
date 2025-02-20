let results = []
for (const i of ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) {
    results.push([document.getElementById('bar'+i), document.getElementById('res'+i)]);
}

let isPainting = false;
let lineWidth = 84;
let startX;
let startY;

let MatrixAdd = (A, B) => A.map((row, i) => B[0].map((_, j) => row.reduce((acc, _, n) => A[i][j] + B[i][j], 0 )));
let MatrixMult = (A, B) => A.map((row, i) => B[0].map((_, j) => row.reduce((acc, _, n) => acc + A[i][n] * B[n][j], 0 )));

let sigmoid = (A) => 1.0 / (1.0 + Math.exp(-A));

const network = fetch("digitreader.json")
    .then((response) => response.json())
    .then((network) => {
        return network;
    });

const feed = async(input) => {
    const net = await network;
    const biases = net['biases'];
    const weights = net['weights'];
    let output = math.matrix(input);
    
    for (i in biases) {
        let b = math.matrix(biases[i]);
        let w = math.matrix(weights[i]);

        let temp = math.multiply(w, output);
        let temp2 = math.add(temp, b);
        output = math.map(temp2, sigmoid);
    }
    console.log(output);
    for (i in output._data) {
        let bar = " ";
        let out = output._data[i] * 100;
        for (let j = 0; j <= 100; j+= 5) {
            if (out >= 5) bar += "█";
            else bar += " ";
            out -= 5;
        }
        results[i][0].innerHTML = bar;
        results[i][1].innerHTML = Math.round(output._data[i]*10000) / 100 + '%';
    }
    return output;
}

const eval = () => {
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    let screen = ctx.getImageData(0, 0, 784, 784);
    let data = [];
    let countX = 0;
    let countY = 0;
    let temp = 0;
    for (i in screen.data) {
        if (i % 4 != 0) continue;
        if (countX % 28 == 0 && countY % 28 == 0) data.push([]);
        if (countX == 784) {
            countX = 0;
            countY += 1;
        }
        data[Math.floor(countX/28) + 28 * Math.floor(countY/28)].push(255 - screen.data[i]);
        countX += 1;
        temp += 1;
    }
    for (i in data) {
        avg = 0;
        for (j in data[i]) {
            avg += data[i][j] / 784;
        }
        data[i] = [Math.round(avg)];
    }
    data.pop(); // temporary bc idk why there is 785 elements and not 784
    feed(data);
}

const saveJson = () => {
    let screen = ctx.getImageData(0, 0, 784, 784);
    let data = [];
    let countX = 0;
    let countY = 0;
    let temp = 0;
    for (i in screen.data) {
        if (i % 4 != 0) continue;
        if (countX % 28 == 0 && countY % 28 == 0) data.push([]);
        if (countX == 784) {
            countX = 0;
            countY += 1;
        }
        data[Math.floor(countX/28) + 28 * Math.floor(countY/28)].push(255 - screen.data[i]);
        countX += 1;
        temp += 1;
    }
    for (i in data) {
        avg = 0;
        for (j in data[i]) {
            avg += data[i][j] / 784;
        }
        data[i] = [Math.round(avg)];
    }
    data.pop(); // temporary bc idk why there is 785 elements and not 784

    let jsonData = "{ \"values\": ["
    for (i in data) {
        jsonData += "[" + data[i] + "]";
        if (i != data.length - 1) jsonData += ',';
    }
    jsonData += "]}";
    console.log(jsonData);
    const name = "data.json";
    const type = "text/plain";

    const a = document.createElement("a");
    const file = new Blob([jsonData], {type: type});
    a.href = URL.createObjectURL(file);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
}


const toolbar = document.getElementById('toolbar');

toolbar.addEventListener('click', e => {
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    if (e.target.id === 'clear') {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (i in results) {
            results[i][0].innerHTML = ' ';
            results[i][1].innerHTML = '0%';
        }
    } else if (e.target.id === 'save') {
        saveJson();
    }
});

const el = document.getElementById('drawing-board');

el.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

el.addEventListener('mouseup', (e) => {
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
    eval();
});

el.addEventListener('mousemove', (e) =>{
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    if(!isPainting) {
        return;
    }
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY);
    ctx.stroke();

});

const ongoingTouches = [];

function copyTouch({ identifier, pageX, pageY }) {
  return { identifier, pageX, pageY };
}

const ongoingTouchIndexById = (idToFind) => {
  for (let i = 0; i < ongoingTouches.length; i++) {
    const id = ongoingTouches[i].identifier;

    if (id === idToFind) {
      return i;
    }
  }
  return -1; // not found
}

el.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const touches = e.changedTouches;
    
    for (let i = 0; i < touches.length; i++) {
        ongoingTouches.push(copyTouch(touches[i]));
        ctx.beginPath();
        console.log(ongoingTouches);
    }
});

el.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const touches = e.changedTouches;

    for (let i = 0; i < touches.length; i++) {
        const idx = ongoingTouchIndexById(touches[i].identifier);
        if (idx >= 0) {
            ctx.beginPath();
            ctx.moveTo(ongoingTouches[idx].pageX, ongoingTouches[idx].pageY);
            ctx.lineTo(touches[i].pageX, touches[i].pageY);
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = "black";
            ctx.lineCap = 'round';
            ctx.stroke();
            ongoingTouches.splice(idx, 1, copyTouch(touches[i]));
        }
    }
});

el.addEventListener('touchend', (e) => {
    e.preventDefault();
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const touches = e.changedTouches;

    for (let i = 0; i < touches.length; i++) {
        let idx = ongoingTouchIndexById(touches[i].identifier);
        if (idx >= 0) {
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(ongoingTouches[idx].pageX, ongoingTouches[idx].pageY);
            ctx.lineTo(touches[i].pageX, touches[i].pageY);
            ongoingTouches.splice(idx, 1);
        }
    }
    eval();
});

el.addEventListener('touchcancel', (e) => {
    e.preventDefault();
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    const touches = e.changedTouches;

    for (let i = 0; i < touches.length; i++) {
        let idx = ongoingTouchIndexById(touches[i].identifier);
        ongoingTouches.splice(idx, 1);
    }
});


document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById('drawing-board');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

