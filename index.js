import  * as tf from '@tensorflow/tfjs-node';
import chalk from 'chalk';
import fs from 'fs';


let snake = [{ top: 3, left: 3 }]; // Snake starts in the middle of our 10x10 grid
let apple = { top: 7, left: 7 }; // Apple starts at the bottom right
let epsilon = 1//0.0001;
let predictDirection;
const delayStep = 200;

let games = 0;
const printEveryNGames = 1;
const gamma = 0.9;  // discount factor for future rewards
let memory = [];  // stores {state, action, reward, nextState} tuples for training

const stateLength = 6;
let previousSnakeHeadPosition = null; // Initialize this wherever the game starts
let newDistanceToFood = null;
let prevDistanceToFood = null;


// Possible directions for the snake to move
const directions = ['top', 'bottom', 'left', 'right'];

// Use TensorFlow.js
let sumReward = 0;
let model;
let modelIsLoaded = false;

if (fs.existsSync('my-model/model.json')) {
    model = await tf.loadLayersModel('file://my-model/model.json');
    modelIsLoaded = true;
} else {
    model = tf.sequential();

    // Add first dense layer with 32 neurons and 'relu' activation function
    model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [stateLength]}));

    // Add another dense layer with 32 neurons and 'relu' activation function
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));

    // Output is 4 for the possible directions the snake can move
    // Use a linear activation function for the output layer
    model.add(tf.layers.dense({units: 4, activation: 'linear'}));

}

// Compile model with 'adam' optimizer instead of 'sgd'
model.compile({optimizer: 'adam', loss: 'meanSquaredError'});

let steps = 0;
let reward = 0;
let time = 0;

async function moveSnake() {
    const oldState = getState();

    let oldStateTensor = await tf.tensor2d(oldState, [1, stateLength]);

    // Add randomness to encourage exploration
    let direction;
    if (/*!modelIsLoaded &&*/ Math.random() < 0.3 - games * epsilon) {  // 30% of the time, move randomly
        direction = Math.floor(Math.random() * 4); // randomly choose a direction
    } else { // 70% of the time, use the model's prediction
        direction = (await tf.argMax(await model.predict(oldStateTensor), 1).array())[0];
    }

    predictDirection = directions[direction];
    reward = await performAction(predictDirection);

    await drawGame(snake);

    sumReward += reward;
    console.log(sumReward < 0 ? chalk.red("sumReward: " + sumReward) : chalk.green("sumReward: " + sumReward));
    console.log("predictDirection: " + predictDirection);
    console.log("Randomness: " + (0.3 - games * epsilon).toFixed(2));
    console.log(reward < 0 ? chalk.red("reward: " + reward) : chalk.green("reward: " + reward));
    console.log("currDistanceToFood: " + newDistanceToFood);
    console.log("prevDistanceToFood: " + prevDistanceToFood);

    time = new Date().getTime();

    if (games % printEveryNGames === 0) {
        while (time > new Date().getTime() - delayStep) {
            void 0;
        }
    }

    prevDistanceToFood = newDistanceToFood;

    console.log('+----------+\n');

    const newState = getState();

    let newStateTensor = await tf.tensor2d(newState, [1, stateLength]);

    let oldQs = await model.predict(oldStateTensor).array();
    let newQs = await model.predict(newStateTensor).array();
    // Print Q values
    // console.log("Current Qs: ", newQs);
    // console.log("Next Q: ", oldQs);
    let updatedReward = reward + gamma * Math.max(...newQs[0]);

    newQs[0][direction] = updatedReward;

    await model.fit(oldStateTensor, await tf.tensor2d(newQs, [1, 4]), {epochs: 5});

    if (reward === -25) {
        await newGame();
    }
    return predictDirection;
}

async function performAction(direction) {
    steps++;
    const head = {...snake[0]}; // copy head

    switch (direction) {
        case 'top':
            head.top--;
            break;
        case 'bottom':
            head.top++;
            break;
        case 'left':
            head.left--;
            break;
        case 'right':
            head.left++;
            break;
    }

    snake.unshift(head); // add new head to snake


    newDistanceToFood = getManhattanDistanceToFood();


    // If the snake hits the boundary or itself, it's game over and the reward is -1
    if (isDanger(head)) {
        return -25;
    }

    // If the snake eats an apple, the reward is 1
    if (head.top === apple.top && head.left === apple.left) {
        apple.top = Math.floor(Math.random() * 10);
        apple.left = Math.floor(Math.random() * 10);
        return 25;
    }

    snake.pop(); // remove tail

    // If the snake moves farther to the apple and has no tail, the reward is -1
    if (prevDistanceToFood && snake.length === 1) {
        if (prevDistanceToFood
            <
            newDistanceToFood
        ) {
            return -1;
        }
        return 1;
    }

    // Otherwise, the snake moves normally and the reward is 0
    return 0;
}


function getState() {
    const foodDir = getFoodDir();

    // Get the new danger direction after the action is performed
    const dangerDir = getDangerDir();

    /*
      const currentPos = [snake[0].top, snake[0].left];

      let direction;
       if (previousSnakeHeadPosition === null) {
           // The game just started, so we don't have a direction yet
           direction = [0, 0];
       } else {
           // Calculate direction as the difference between the current and previous position
           direction = [
               currentPos[0] - previousSnakeHeadPosition[0],
               currentPos[1] - previousSnakeHeadPosition[1]
           ];
       }

       previousSnakeHeadPosition = currentPos; // Update the previous position
       */

    let avgTailPos = [0, 0];
    if(snake.length > 1) {
        for(let i = 1; i < snake.length; i++) {
            avgTailPos[0] += snake[i].top;
            avgTailPos[1] += snake[i].left;
        }
        avgTailPos = [avgTailPos[0] / (snake.length - 1), avgTailPos[1] / (snake.length - 1)];
    } else {
        // Handle case where there's no tail (snake.length equals one)
        // You can set avgTailPos to the position of the head, or any other value that makes sense in your specific situation
        avgTailPos = [snake[0].top, snake[0].left];
    }

    // Get the middle segment of the tail
    const midSegmentIndex = Math.floor(snake.length / 2);
    const midSegmentPos = snake[midSegmentIndex] ? [snake[midSegmentIndex].top, snake[midSegmentIndex].left] : [0, 0];

    // Get the segment a quarter way down the tail
    const quarterSegmentIndex = Math.floor(snake.length / 4);
    const quarterSegmentPos = snake[quarterSegmentIndex] ? [snake[quarterSegmentIndex].top, snake[quarterSegmentIndex].left] : [0, 0];

    // Get the segment three quarters down the tail
    const threeQuartersSegmentIndex = Math.floor(snake.length * 3 / 4);
    const threeQuartersSegmentPos = snake[threeQuartersSegmentIndex] ? [snake[threeQuartersSegmentIndex].top, snake[threeQuartersSegmentIndex].left] : [0, 0];

    // Get the last segment of the tail (end of the tail)
    const endSegmentPos = snake[snake.length - 1] ? [snake[snake.length - 1].top, snake[snake.length - 1].left] : [0, 0];

    // Get the new distance to food after the action is performed

    // The new state of the game
    const nextState = [/*...direction,*/ ...dangerDir, ...foodDir, /*getManhattanDistanceToFood(), ...avgTailPos*//*, ...midSegmentPos, ...quarterSegmentPos, ...threeQuartersSegmentPos, ...endSegmentPos*/];

    return nextState; // Array(9) [0,1,0, 0,1,  -1, 6,9,3]
}




function sampleBatch(memory, batchSize) {
    const batch = [];

    for (let i = 0; i < batchSize; i++) {
        const index = Math.floor(Math.random() * memory.length);
        batch.push(memory[index]);
    }

    return batch;
}

function drawGame(snake) {
    if (games % printEveryNGames === 0) {
        console.clear();

        // print the top border
        console.log('+--------------------+');

        for (let i = 0; i < 10; i++) {
            let line = '|'; // Add the side border

            for (let j = 0; j < 10; j++) {
                let segmentType = null;
                for (let k = 0; k < snake.length; k++) {
                    if (snake[k].top === i && snake[k].left === j) {
                        // Check if current segment is the head or part of the tail
                        segmentType = k === 0 ? chalk.green('🤩') : chalk.blue('🦠');
                        break;
                    }
                }

                if (apple.top === i && apple.left === j) {
                    line += chalk.red(  '🍎');
                } else if (segmentType) {
                    line += segmentType;
                } else {
                    line += '  ';
                }
            }
            line += '|'; // Add the side border
            console.log(line);
        }

        // print the bottom border
        console.log('+--------------------+');
    }
}





function getDangerDir() {
    const directions = ['up', 'down', 'left', 'right'];
    return directions.map(dir => isDanger(getNextPosition(dir)) ? 1 : 0);
}


function getNextPosition(direction) {
    // Get the current position of the snake's head
    const currentPos = {
        top: snake[0].top,
        left: snake[0].left
    };

    // Calculate the next position based on the current direction
    switch(direction) {
        case 'up':
            currentPos.top -= 1;
            break;
        case 'down':
            currentPos.top += 1;
            break;
        case 'left':
            currentPos.left -= 1;
            break;
        case 'right':
            currentPos.left += 1;
            break;
        default:
        // Stay still if the direction is unknown
    }

    return currentPos;
}


function isDanger(position) {
    // Check if the given position is out of bounds or overlaps with the snake.
    return isOutOfBounds(position) || isOnSnake(position);
}

function isOutOfBounds(position) {
    return position.top < 0 || position.top >= 10 || position.left < 0 || position.left >= 10;
}

function isOnSnake(position) {
   /* return snake.some((segment, index) => {
        if (excludeHead && index === 0) return false;
        return segment.top === position.top && segment.left === position.left;
    });*/


    for (let i = 0; i < snake.length; i++) {
        if (i === 0) continue;
        if (snake[i].top === position.top && snake[i].left === position.left) {
            return true;
        }
    }
    return false;
}


function getFoodDir() {
    // Calculate the difference in the x and y coordinates
    // between the apple and the snake's head.
    // If the result is negative, the apple is to the left (for x) or up (for y).
    // If the result is positive, the apple is to the right (for x) or down (for y).
    const directionToFood = {
        top: apple.top - snake[0].top,
        left: apple.left - snake[0].left
    };

    console.log("x:", directionToFood.left < 0 ? "left" : directionToFood.left > 0 ? "right" : "vertical")
    console.log("y:", directionToFood.top < 0 ? "above" : directionToFood.top > 0 ? "below" : "horizontal")
    return [
        // For the x direction:
        // If the apple is to the left, return -1.
        // If the apple is to the right, return 1.
        // If the apple is on the same vertical line as the snake's head, return 0.
        directionToFood.left < 0 ? -1 : directionToFood.left > 0 ? 1 : 0,

        // For the y direction:
        // If the apple is above, return -1.
        // If the apple is below, return 1.
        // If the apple is on the same horizontal line as the snake's head, return 0.
        directionToFood.top < 0 ? -1 : directionToFood.top > 0 ? 1 : 0
    ];
}




function getEuclideanDistanceToFood() {
    const dx = apple.left - snake[0].left;
    const dy = apple.top - snake[0].top;
    return Math.sqrt(dx * dx + dy * dy);
}

function getManhattanDistanceToFood() {
    const dx = Math.abs(apple.left - snake[0].left);
    const dy = Math.abs(apple.top - snake[0].top);
    return dx + dy;
}



async function newGame() {
    games++;
    steps = 0;
    sumReward = 0;
    snake = [{ top: 3, left: 3 }]; // Snake starts in the middle of our 10x10 grid
    apple = { top: 7, left: 7 }; // Apple starts at the bottom right
}

async function gameLoop() {
    while(true) {
        predictDirection = await moveSnake();
        if (games % printEveryNGames === 0 ) {
           await model.save('file://my-model');
           // await sleep(250);
        }
        else {
            // await sleep(30);
        }

    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

(async ()=> {
    await newGame()
    await gameLoop();
})();

