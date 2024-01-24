package rbs.transformer;
//
//
// import seakers.vassar.problems.Assigning.GigaAssigningParams;
// import seakers.vassar.BaseParams;
// import seakers.vassar.evaluation.ArchitectureEvaluationManager;
// import seakers.vassar.problems.Assigning.ArchitectureEvaluator;
// import seakers.vassartest.search.problems.Assigning.AssigningArchitecture;
// import seakers.vassartest.search.problems.Assigning.AssigningProblem;
// import seakers.vassartest.search.problems.Assigning.GigaArchitecture;
// import seakers.vassar.Result;
// import java.util.concurrent.CompletableFuture;
//
//
// import java.io.File;
// import java.io.FileWriter;
// import java.io.IOException;
// import java.util.*;
// import java.util.concurrent.*;
// import java.util.ArrayList;
//
// public class EvaluationGigaAssigning {
//
//     public GigaAssigningParams params;
//     public ArchitectureEvaluator evaluator;
//     public ArchitectureEvaluationManager evaluationManager;
//     public AssigningProblem problem;
//
//     public HashMap<String, Future<Result>> cache;
//
//
//     public EvaluationGigaAssigning() {
//         System.out.println("--> TESTING");
//         int orekit_threads = 1;
//
//         // Problem Parameters
//         String resourcesPath = "/home/ubuntu/vassar/giga/VASSAR_resources";
//         this.params = new GigaAssigningParams(resourcesPath, "FUZZY-CASES", "test", "normal", orekit_threads);
//
//         // Evaluator
//         this.evaluator = new ArchitectureEvaluator();
//
//         // Manager
//         int numCpus = 5;
//         this.evaluationManager = new ArchitectureEvaluationManager(params, evaluator);
//         evaluationManager.init(numCpus);
//
//         // Problem
//         this.problem = new AssigningProblem(new int[]{1}, "GigaProblem", evaluationManager, params);
//
//         // Cache
//         this.cache = new HashMap<>();
//
//     }
//
//
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
//
//
//
//
//
// }