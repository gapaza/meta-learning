package rbs.transformer;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.core.*;
import org.moeaframework.problem.AbstractProblem;
import com.mathworks.engine.*;
import java.util.concurrent.*;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;
import seakers.trussaos.problems.GabeTrussProblem;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryVariable;
import java.util.ArrayList;


public class EvaluationTruss {

    public AbstractProblem trussProblem;
    public MatlabEngine engine = null;
    public String csvPath;

    public EvaluationTruss(){

    }

     public void initProblem2(double memberRadius, double sideLength, double yModulus){

        // ------------------------------
        // Model Choice
        // ------------------------------

        /**
         * modelChoice = 0 --> Fibre Stiffness Model
         *             = 1 --> Truss Stiffness Model
         *             = 2 --> Beam Model
         */
        int modelChoice = 1; // Fibre stiffness model cannot be used for the artery problem
        String csvPath = "/Users/gapaza/repos/seakers/truss/KDDMM/Truss_AOS";

        boolean arteryProblem = false; // Solve the artery optimization (otherwise the original truss problem is solved)
        boolean useOptimizationProblem2 = true; // Use ConstantRadiusTrussProblem2 as problem class (instead of ConstantRadiusTrussProblem)
        double targetStiffnessRatio = 1;
        if (arteryProblem) {
            targetStiffnessRatio = 0.421;
        }

        // ------------------------------
        // Heuristics
        // ------------------------------

        boolean[] partialCollapsibilityConstrained = {false, false, false, false, false, false, false};
        boolean[] nodalPropertiesConstrained = {false, false, false, false, false, false, false};
        boolean[] orientationConstrained = {false, false, false, false, false, false, false};
        boolean[] intersectionConstrained = {false, true, false, false, false, false, false};

        boolean[][] heuristicsConstrained = new boolean[4][7];
        for (int i = 0; i < 7; i++) {
            heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
            heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
            heuristicsConstrained[2][i] = orientationConstrained[i];
            heuristicsConstrained[3][i] = intersectionConstrained[i];
        }
        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        for (int i = 0; i < 4; i++) {
            if (heuristicsConstrained[i][5]) {
                numberOfHeuristicConstraints++;
            }
            if (heuristicsConstrained[i][4]) {
                numberOfHeuristicObjectives++;
            }
        }

        try {
            // If matlab engine is not null, close it
            if (this.engine != null) {
                System.out.println("Closing MATLAB engine...");
                this.engine.close();
                System.out.println("MATLAB engine closed.");
            }

            System.out.println("Starting MATLAB engine...");
            String myPath = "/Users/gapaza/repos/seakers/truss/KDDMM/Truss_AOS";
            this.engine = MatlabEngine.startMatlab();
            this.engine.eval("addpath('" + myPath + "')", null, null);
            this.engine.eval("warning('off', 'MATLAB:singularMatrix');");
            System.out.println("MATLAB engine started.");
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }

        // ------------------------------
        // Problem Settings
        // ------------------------------

        double printableRadius = memberRadius;  // 250e-6; // in m
        double printableSideLength = sideLength;  // 10e-3; // in m
        double printableModulus = yModulus;  // 1.8162e6; // in Pa
        double sideNodeNumber = 3.0D;
        System.out.println(" | Radius: " + printableRadius + " | Side: " + printableSideLength + " Modulus: " + printableModulus);

        // ------------------------------
        // Number of Variables
        // ------------------------------

        int nucFactor = 3; // Not used if PBC model is used
        int totalNumberOfMembers;
        if (sideNodeNumber >= 5) {
            int sidenumSquared = (int) (sideNodeNumber*sideNodeNumber);
            totalNumberOfMembers =  sidenumSquared * (sidenumSquared - 1)/2;
        }
        else {
            totalNumberOfMembers = (int) (CombinatoricsUtils.factorial((int) (sideNodeNumber*sideNodeNumber))/(CombinatoricsUtils.factorial((int) ((sideNodeNumber*sideNodeNumber) - 2)) * CombinatoricsUtils.factorial(2)));
        }
        int numberOfRepeatableMembers = (int) (2 * (CombinatoricsUtils.factorial((int) sideNodeNumber)/(CombinatoricsUtils.factorial((int) (sideNodeNumber - 2)) * CombinatoricsUtils.factorial(2))));
        int numVariables = totalNumberOfMembers - numberOfRepeatableMembers;

        // ------------------------------
        // Init Problem
        // ------------------------------

        if (arteryProblem) {
            this.trussProblem = new ConstantRadiusArteryProblem(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, heuristicsConstrained);
        } else {
            if (useOptimizationProblem2) {
                this.trussProblem = new GabeTrussProblem(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, heuristicsConstrained);
            } else {
                this.trussProblem = new ConstantRadiusTrussProblem(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, partialCollapsibilityConstrained[0], nodalPropertiesConstrained[0], orientationConstrained[0]);
            }
        }
    }




















    // Init Problem
    public ArrayList<Double> initProblem(int config_num, boolean run_val){

        // ------------------------------
        // Model Choice
        // ------------------------------

        /**
         * modelChoice = 0 --> Fibre Stiffness Model
         *             = 1 --> Truss Stiffness Model
         *             = 2 --> Beam Model
         */
        int modelChoice = 1; // Fibre stiffness model cannot be used for the artery problem
        String csvPath = "/Users/gapaza/repos/seakers/truss/KDDMM/Truss_AOS";

        boolean arteryProblem = false; // Solve the artery optimization (otherwise the original truss problem is solved)
        boolean useOptimizationProblem2 = true; // Use ConstantRadiusTrussProblem2 as problem class (instead of ConstantRadiusTrussProblem)
        double targetStiffnessRatio = 1;
        if (arteryProblem) {
            targetStiffnessRatio = 0.421;
        }

        // ------------------------------
        // Heuristics
        // ------------------------------

        boolean[] partialCollapsibilityConstrained = {false, false, false, false, false, false, false};
        boolean[] nodalPropertiesConstrained = {false, false, false, false, false, false, false};
        boolean[] orientationConstrained = {false, false, false, false, false, false, false};
        boolean[] intersectionConstrained = {false, true, false, false, false, false, false};

        boolean[][] heuristicsConstrained = new boolean[4][7];
        for (int i = 0; i < 7; i++) {
            heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
            heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
            heuristicsConstrained[2][i] = orientationConstrained[i];
            heuristicsConstrained[3][i] = intersectionConstrained[i];
        }
        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        for (int i = 0; i < 4; i++) {
            if (heuristicsConstrained[i][5]) {
                numberOfHeuristicConstraints++;
            }
            if (heuristicsConstrained[i][4]) {
                numberOfHeuristicObjectives++;
            }
        }

        try {
            // If matlab engine is not null, close it
            if (this.engine != null) {
                System.out.println("Closing MATLAB engine...");
                this.engine.close();
                System.out.println("MATLAB engine closed.");
            }

            System.out.println("Starting MATLAB engine...");
            String myPath = "/Users/gapaza/repos/seakers/truss/KDDMM/Truss_AOS";
            this.engine = MatlabEngine.startMatlab();
            this.engine.eval("addpath('" + myPath + "')", null, null);
            this.engine.eval("warning('off', 'MATLAB:singularMatrix');");
            System.out.println("MATLAB engine started.");
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }


        // ------------------------------
        // Validation Settings
        // ------------------------------

        double printableRadius = 250e-6; // in m
        double printableSideLength = 10e-3; // in m
        double printableModulus = 1.8162e6; // in Pa
        double sideNodeNumber = 3.0D;

        // ------------------------------
        // Train Settings
        // ------------------------------

        if (!run_val) {
            // Create list of configs, with each config being a list of 2 values: printableRadius, printableSideLength
            ArrayList<ArrayList<Double>> configs = new ArrayList<ArrayList<Double>>();
            ArrayList<Double> rad_values = new ArrayList<Double>();
//             rad_values.add(250e-6);
            rad_values.add(500e-6);
            rad_values.add(750e-6);
            rad_values.add(1000e-6);
            rad_values.add(1250e-6);
            ArrayList<Double> side_values = new ArrayList<Double>();
//             side_values.add(10e-3);
            side_values.add(20e-3);
            side_values.add(30e-3);
            side_values.add(40e-3);
            side_values.add(50e-3);
            // Enumerate all possible configs
            for(Double rad: rad_values){
                for(Double side: side_values){
                    ArrayList<Double> config = new ArrayList<Double>();
                    config.add(rad);
                    config.add(side);
                    configs.add(config);
                }
            }
            // Get config
            ArrayList<Double> config = configs.get(config_num);
            printableRadius = config.get(0);
            printableSideLength = config.get(1);
        }

        ArrayList<Double> return_configs = new ArrayList<Double>();
        return_configs.add(printableRadius);
        return_configs.add(printableSideLength);
        System.out.println("Config: " + config_num + " | Radius: " + printableRadius + " | Side: " + printableSideLength);

        int nucFactor = 3; // Not used if PBC model is used
        int totalNumberOfMembers;
        if (sideNodeNumber >= 5) {
            int sidenumSquared = (int) (sideNodeNumber*sideNodeNumber);
            totalNumberOfMembers =  sidenumSquared * (sidenumSquared - 1)/2;
        }
        else {
            totalNumberOfMembers = (int) (CombinatoricsUtils.factorial((int) (sideNodeNumber*sideNodeNumber))/(CombinatoricsUtils.factorial((int) ((sideNodeNumber*sideNodeNumber) - 2)) * CombinatoricsUtils.factorial(2)));
        }
        int numberOfRepeatableMembers = (int) (2 * (CombinatoricsUtils.factorial((int) sideNodeNumber)/(CombinatoricsUtils.factorial((int) (sideNodeNumber - 2)) * CombinatoricsUtils.factorial(2))));
        int numVariables = totalNumberOfMembers - numberOfRepeatableMembers;

        double[][] globalNodePositions;
        if (arteryProblem) {
            this.trussProblem = new ConstantRadiusArteryProblem(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, heuristicsConstrained);
            globalNodePositions = ((ConstantRadiusArteryProblem) this.trussProblem).getNodalConnectivityArray();
        } else {
            if (useOptimizationProblem2) {
                this.trussProblem = new ConstantRadiusTrussProblem2(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, heuristicsConstrained);
                globalNodePositions = ((ConstantRadiusTrussProblem2) this.trussProblem).getNodalConnectivityArray();
            } else {
                this.trussProblem = new ConstantRadiusTrussProblem(csvPath, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, this.engine, partialCollapsibilityConstrained[0], nodalPropertiesConstrained[0], orientationConstrained[0]);
                globalNodePositions = ((ConstantRadiusTrussProblem) this.trussProblem).getNodalConnectivityArray();
            }
        }

        return return_configs;
    }




    public ArrayList<Double> evaluateDesign(ArrayList<Double> design){
        Solution newSol = new Solution(this.trussProblem.getNumberOfVariables(), this.trussProblem.getNumberOfObjectives(),this.trussProblem.getNumberOfConstraints());
        for (int i = 0; i < this.trussProblem.getNumberOfVariables(); i++) {
            boolean bit = design.get(i) > 0.5;
            BinaryVariable newVar = new BinaryVariable(1);
            newVar.set(0, bit);
            newSol.setVariable(i, newVar);
        }
//         System.out.println("Evaluating design...");
        this.trussProblem.evaluate(newSol);
//         System.out.println("Design evaluated.");
        ArrayList<Double> result = new ArrayList<Double>();
        result.add((Double) newSol.getAttribute("TrueObjective1"));
        result.add((Double) newSol.getAttribute("TrueObjective2"));
//         System.out.println("True Objective 1: " + result.get(0));
//         System.out.println("True Objective 2: " + result.get(1));
        return result;
    }


//     public synchronized Future<Result> evaluateDesignAsync(ArrayList<Double> design){
//
//         String bit_string = "";
//         int counter = 0;
//         for(Double d: design){
//             if(d == 1.0){
//                 counter++;
//             }
//             bit_string += Integer.toString(d.intValue());
//         }
//
//         // Max Instrument Constraint
//         if(counter > 35 || counter == 0){
//             System.out.println("--> INFEASIBLE INSTRUMENT COUNT: " + counter + " instruments");
//             Result result = new Result(null, 0.0,  35000);
//             return CompletableFuture.completedFuture(result);
//         }
//
//         // Cache
// //         if(this.cache.containsKey(bit_string)){
// //             System.out.println("--> DESIGN EXISTS: " + counter + " instruments");
// //             return this.cache.get(bit_string);
// //         }
// //         else{
// //             System.out.println("-----> NEW DESIGN: " + counter + " instruments");
// //             GigaArchitecture arch = new GigaArchitecture(bit_string);
// //             Future<Result> future_result = this.problem.evaluateGigaArchAsync(arch);
// //             this.cache.put(bit_string, future_result);
// //             return future_result;
// //         }
//
//         // No Cache
//         System.out.println("-----> DESIGN: " + counter + " instruments");
//         GigaArchitecture arch = new GigaArchitecture(bit_string);
//         Future<Result> future_result = this.problem.evaluateGigaArchAsync(arch);
//         this.cache.put(bit_string, future_result);
//         return future_result;
//     }





}