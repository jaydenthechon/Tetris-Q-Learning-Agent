// package src.pas.tetris.agents;


// // SYSTEM IMPORTS
// import java.util.Iterator;
// import java.util.List;
// import java.util.Random;


// // JAVA PROJECT IMPORTS
// import edu.bu.tetris.agents.QAgent;
// import edu.bu.tetris.agents.TrainerAgent.GameCounter;
// import edu.bu.tetris.game.Board;
// import edu.bu.tetris.game.Game.GameView;
// import edu.bu.tetris.game.minos.Mino;
// import edu.bu.tetris.linalg.Matrix;
// import edu.bu.tetris.nn.Model;
// import edu.bu.tetris.nn.LossFunction;
// import edu.bu.tetris.nn.Optimizer;
// import edu.bu.tetris.nn.models.Sequential;
// import edu.bu.tetris.nn.layers.Dense; // fully connected layer
// import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
// import edu.bu.tetris.nn.layers.Tanh;
// import edu.bu.tetris.nn.layers.Sigmoid;
// import edu.bu.tetris.training.data.Dataset;
// import edu.bu.tetris.utils.Pair;


// public class TetrisQAgent
//     extends QAgent
// {

//     public static final double EXPLORATION_PROB = 0.05;

//     private Random random;

//     public TetrisQAgent(String name)
//     {
//         super(name);
//         this.random = new Random(12345); // optional to have a seed
//     }

//     public Random getRandom() { return this.random; }

//     @Override
//     public Model initQFunction()
//     {
//         // System.out.println("initQFunction called!");
//         // build a single-hidden-layer feedforward network
//         // this example will create a 3-layer neural network (1 hidden layer)
//         // in this example, the input to the neural network is the
//         // image of the board unrolled into a giant vector
//         final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
//         final int hiddenDim = 2 * numPixelsInImage;
//         final int outDim = 1;

//         Sequential qFunction = new Sequential();
//         qFunction.add(new Dense(numPixelsInImage, hiddenDim));
//         qFunction.add(new Tanh());
//         qFunction.add(new Dense(hiddenDim, outDim));

//         return qFunction;
//     }

//     /**
//         This function is for you to figure out what your features
//         are. This should end up being a single row-vector, and the
//         dimensions should be what your qfunction is expecting.
//         One thing we can do is get the grayscale image
//         where squares in the image are 0.0 if unoccupied, 0.5 if
//         there is a "background" square (i.e. that square is occupied
//         but it is not the current piece being placed), and 1.0 for
//         any squares that the current piece is being considered for.
        
//         We can then flatten this image to get a row-vector, but we
//         can do more than this! Try to be creative: how can you measure the
//         "state" of the game without relying on the pixels? If you were given
//         a tetris game midway through play, what properties would you look for?
//      */
//     @Override
//     public Matrix getQFunctionInput(final GameView game,
//                                     final Mino potentialAction)
//     {
//         Matrix flattenedImage = null;
//         try
//         {
//             flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
//         } catch(Exception e)
//         {
//             e.printStackTrace();
//             System.exit(-1);
//         }
//         return flattenedImage;
//     }

//     /**
//      * This method is used to decide if we should follow our current policy
//      * (i.e. our q-function), or if we should ignore it and take a random action
//      * (i.e. explore).
//      *
//      * Remember, as the q-function learns, it will start to predict the same "good" actions
//      * over and over again. This can prevent us from discovering new, potentially even
//      * better states, which we want to do! So, sometimes we should ignore our policy
//      * and explore to gain novel experiences.
//      *
//      * The current implementation chooses to ignore the current policy around 5% of the time.
//      * While this strategy is easy to implement, it often doesn't perform well and is
//      * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
//      * strategy here.
//      */
//     @Override
//     public boolean shouldExplore(final GameView game,
//                                  final GameCounter gameCounter)
//     {
//         // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
//         return this.getRandom().nextDouble() <= EXPLORATION_PROB;
//     }

//     /**
//      * This method is a counterpart to the "shouldExplore" method. Whenever we decide
//      * that we should ignore our policy, we now have to actually choose an action.
//      *
//      * You should come up with a way of choosing an action so that the model gets
//      * to experience something new. The current implemention just chooses a random
//      * option, which in practice doesn't work as well as a more guided strategy.
//      * I would recommend devising your own strategy here.
//      */
//     @Override
//     public Mino getExplorationMove(final GameView game)
//     {
//         int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
//         return game.getFinalMinoPositions().get(randIdx);
//     }

//     /**
//      * This method is called by the TrainerAgent after we have played enough training games.
//      * In between the training section and the evaluation section of a phase, we need to use
//      * the exprience we've collected (from the training games) to improve the q-function.
//      *
//      * You don't really need to change this method unless you want to. All that happens
//      * is that we will use the experiences currently stored in the replay buffer to update
//      * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
//      * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
//      * (i.e. all at once)...this often works better and is an active area of research.
//      *
//      * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
//      * of epochs in between the training and eval sections of each phase.
//      */
//     @Override
//     public void trainQFunction(Dataset dataset,
//                                LossFunction lossFunction,
//                                Optimizer optimizer,
//                                long numUpdates)
//     {
//         for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
//         {
//             dataset.shuffle();
//             Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

//             while(batchIterator.hasNext())
//             {
//                 Pair<Matrix, Matrix> batch = batchIterator.next();

//                 try
//                 {
//                     Matrix YHat = this.getQFunction().forward(batch.getFirst());

//                     optimizer.reset();
//                     this.getQFunction().backwards(batch.getFirst(),
//                                                   lossFunction.backwards(YHat, batch.getSecond()));
//                     optimizer.step();
//                 } catch(Exception e)
//                 {
//                     e.printStackTrace();
//                     System.exit(-1);
//                 }
//             }
//         }
//     }

//     /**
//      * This method is where you will devise your own reward signal. Remember, the larger
//      * the number, the more "pleasurable" it is to the model, and the smaller the number,
//      * the more "painful" to the model.
//      *
//      * This is where you get to tell the model how "good" or "bad" the game is.
//      * Since you earn points in this game, the reward should probably be influenced by the
//      * points, however this is not all. In fact, just using the points earned this turn
//      * is a **terrible** reward function, because earning points is hard!!
//      *
//      * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
//      * of the game. For instance, the higher the stack of minos gets....generally the worse
//      * (unless you have a long hole waiting for an I-block). When you design a reward
//      * signal that is less sparse, you should see your model optimize this reward over time.
//      */
//     @Override
//     public double getReward(final GameView game)
//     {
//         return game.getScoreThisTurn();
//     }

// }
package src.pas.tetris.agents;

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Block;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // activation functions
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;

public class TetrisQAgent extends QAgent {
    public static final double EXPLORATION_PROB = 0.05;
    private Random random;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345); // optional seed
    }

    public Random getRandom() {
        return this.random;
    }

    @Override
    public Model initQFunction() {
        // Now we have 6 features: maxHeight, sumHeights, numHoles, bumpiness,
        // linesCleared, and immediateScore.
        final int inputDim = 6; 
        final int hiddenDim = 10;
        final int outDim = 1;
        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hiddenDim));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, outDim));
        return qFunction;
    }

    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        // Simulate the board after placing the potential action
        Board simulatedBoard = simulateBoardState(game, potentialAction);

        double[] heights = getHeights(simulatedBoard);
        double maxHeight = getMaxHeight(heights);
        double sumHeights = getSumHeights(heights);
        double numHoles = getNumHoles(simulatedBoard, heights);
        double bumpiness = getBumpiness(heights);
        int linesCleared = getLinesCleared(simulatedBoard);
        double immediateScore = computeImmediateScore(simulatedBoard, linesCleared);

        // 6 features:
        // 0: maxHeight
        // 1: sumHeights
        // 2: numHoles
        // 3: bumpiness
        // 4: linesCleared
        // 5: immediateScore (based on lines cleared and perfect clear)
        Matrix features = Matrix.zeros(1, 6);
        features.set(0, 0, maxHeight);
        features.set(0, 1, sumHeights);
        features.set(0, 2, numHoles);
        features.set(0, 3, bumpiness);
        features.set(0, 4, linesCleared);
        features.set(0, 5, immediateScore);

        return features;
    }

    @Override
    public double getReward(final GameView game) {
        // Compute reward based on the current game state
        Board board = game.getBoard();
        double[] heights = getHeights(board);
        double numHoles = getNumHoles(board, heights);
        double holeDensity = numHoles / (5.0 * 20.0);
        double maxHeight = getMaxHeight(heights);
        double stackHeight = maxHeight / 20.0;
        double tetrisWellDepth = getTetrisWellDepth(heights) / 20.0;
        double efficiency = getEfficiency(game);
        int linesCleared = getLinesCleared(board);

        int score = game.getScoreThisTurn();

        // Alive is 1 if game not over, 0 otherwise
        double alive = game.didAgentLose() ? 0.0 : 1.0;

        // Reward function example:
        double reward = alive * 0.7 
                      + linesCleared * 0.5 
                      + holeDensity * -0.3 
                      + stackHeight * -0.5
                      + tetrisWellDepth * 0.4 
                      + efficiency * 0.5
                      + score * 4;
        if (reward == 0) {
            reward = 0.01;
        }
        return reward;
    }

    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        // Implementing an epsilon-greedy strategy with decaying epsilon
        double epsilon = Math.max(0.1, EXPLORATION_PROB * (1 - gameCounter.getCurrentGameIdx() / 10000.0));
        return this.getRandom().nextDouble() <= epsilon;
    }

    @Override
    public Mino getExplorationMove(final GameView game) {
        // Choose a random action from possible final positions
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
    }

    @Override
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();
            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();
                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());
                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(), lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    // ---------------- HELPER METHODS ----------------

    private Board simulateBoardState(GameView game, Mino potentialAction) {
        // Create a copy of the board and place the Mino onto it,
        // then clear any full lines.
        Board original = game.getBoard();
        Board copy = new Board(original); // Assuming copy constructor
        copy.addMino(potentialAction);
        copy.clearFullLines(); 
        return copy;
    }

    private double[] getHeights(Board board) {
        Block[][] grid = board.getBoard();
        int numRows = Board.NUM_ROWS;
        int numCols = Board.NUM_COLS;

        double[] heights = new double[numCols];
        for (int col = 0; col < numCols; col++) {
            int height = 0;
            for (int row = 0; row < numRows; row++) {
                if (grid[row][col] != null) {
                    height = numRows - row;
                    break;
                }
            }
            heights[col] = height;
        }
        return heights;
    }

    private double getMaxHeight(double[] heights) {
        double maxHeight = 0;
        for (double h : heights) {
            if (h > maxHeight) {
                maxHeight = h;
            }
        }
        return maxHeight;
    }

    private double getSumHeights(double[] heights) {
        double sum = 0;
        for (double h : heights) {
            sum += h;
        }
        return sum;
    }

    private double getNumHoles(Board board, double[] heights) {
        Block[][] grid = board.getBoard();
        int numRows = Board.NUM_ROWS;
        int numCols = Board.NUM_COLS;
        double numHoles = 0;

        for (int col = 0; col < numCols; col++) {
            boolean blockFound = false;
            for (int row = 0; row < numRows; row++) {
                if (grid[row][col] != null) {
                    blockFound = true;
                } else if (blockFound && grid[row][col] == null) {
                    // Empty cell below a block is a hole
                    numHoles += 1;
                }
            }
        }
        return numHoles;
    }

    private double getBumpiness(double[] heights) {
        double bumpiness = 0;
        for (int col = 0; col < heights.length - 1; col++) {
            bumpiness += Math.abs(heights[col] - heights[col + 1]);
        }
        return bumpiness;
    }

    private int getLinesCleared(Board board) {
        Block[][] grid = board.getBoard();
        int numRows = Board.NUM_ROWS;
        int numCols = Board.NUM_COLS;
        int linesCleared = 0;
        for (int row = 0; row < numRows; row++) {
            boolean fullLine = true;
            for (int col = 0; col < numCols; col++) {
                if (grid[row][col] == null) {
                    fullLine = false;
                    break;
                }
            }
            if (fullLine) {
                linesCleared++;
            }
        }
        return linesCleared;
    }

    private double getTetrisWellDepth(double[] heights) {
        int numCols = heights.length;
        double maxWellDepth = 0;
        for (int col = 0; col < numCols; col++) {
            double leftHeight = (col == 0) ? 20 : heights[col - 1];
            double rightHeight = (col == numCols - 1) ? 20 : heights[col + 1];
            double currentHeight = heights[col];
            if (currentHeight < leftHeight - 1 && currentHeight < rightHeight - 1) {
                double wellDepth = Math.min(leftHeight, rightHeight) - currentHeight;
                if (wellDepth > maxWellDepth) {
                    maxWellDepth = wellDepth;
                }
            }
        }
        return maxWellDepth;
    }

    private double getEfficiency(GameView game) {
        // For simplicity, set efficiency to the points earned this turn
        // i.e. how many lines cleared last turn
        return game.getScoreThisTurn();
    }

    private double computeImmediateScore(Board board, int linesCleared) {
        double score = 0.0;

        // Simple scoring scheme:
        // 1 line => ~0.5 points
        // 2 lines => 1 point total
        // 3 lines => 2 points total
        // 4 lines (tetris) => 4 points total
        if (linesCleared == 1) {
            score = 0.5;
        } else if (linesCleared == 2) {
            score = 1.0;
        } else if (linesCleared == 3) {
            score = 2.0;
        } else if (linesCleared == 4) {
            score = 4.0;
        }

        // Check for perfect clear (empty board)
        if (isPerfectClear(board)) {
            score += 6.0;
        }

        // T-spins are not implemented here, but you could add logic if needed.

        return score;
    }

    private boolean isPerfectClear(Board board) {
        Block[][] grid = board.getBoard();
        for (int r = 0; r < Board.NUM_ROWS; r++) {
            for (int c = 0; c < Board.NUM_COLS; c++) {
                if (grid[r][c] != null) {
                    return false;
                }
            }
        }
        return true;
    }
}
