package rbs.transformer;
//
// import org.moeaframework.core.*;
// import org.moeaframework.core.variable.BinaryVariable;
// import seakers.vassarexecheur.search.problems.assigning.AssigningArchitecture;
// import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
// import java.io.File;
// import java.io.FileWriter;
// import java.io.IOException;
// import java.util.*;
// import java.util.concurrent.*;
// import java.util.ArrayList;
// import seakers.vassarheur.BaseParams;
// import seakers.vassarheur.evaluation.ArchitectureEvaluationManager;
// import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;
// import seakers.vassarheur.problems.Assigning.ClimateCentricAssigningParams;
// import seakers.vassarheur.Result;
//
//
// public class EvaluationAssigningParallel {
//
//     public AssigningProblem l_problem;
//     public ClimateCentricAssigningParams l_params;
//
//     // Constructor
//     public EvaluationAssigningParallel() {
//         int numCpus = 80;
//
//         ExecutorService pool = Executors.newFixedThreadPool(numCpus);
//         CompletionService<Algorithm> ecs = new ExecutorCompletionService<>(pool);
//         boolean[] dutyCycleConstrained = {false, false, false, false, false, false};
//         boolean[] instrumentOrbitRelationsConstrained = {false, false, false, false, false, false};
//         boolean[] interferenceConstrained = {false, false, false, false, false, false};
//         boolean[] packingEfficiencyConstrained = {false, false, false, false, false, false};
//         boolean[] spacecraftMassConstrained = {false, false, false, false, false, false};
//         boolean[] synergyConstrained = {false, false, false, false, false, false};
//         boolean[] instrumentCountConstrained = {false, false, false, false, false, false};
//         boolean[][] heuristicsConstrained = new boolean[7][6];
//         for (int i = 0; i < heuristicsConstrained[0].length; i++) {
//             heuristicsConstrained[0][i] = dutyCycleConstrained[i];
//             heuristicsConstrained[1][i] = instrumentOrbitRelationsConstrained[i];
//             heuristicsConstrained[2][i] = interferenceConstrained[i];
//             heuristicsConstrained[3][i] = packingEfficiencyConstrained[i];
//             heuristicsConstrained[4][i] = spacecraftMassConstrained[i];
//             heuristicsConstrained[5][i] = synergyConstrained[i];
//             heuristicsConstrained[6][i] = instrumentCountConstrained[i];
//         }
//         int numberOfHeuristicConstraints = 0;
//         int numberOfHeuristicObjectives = 0;
//         for (int i = 0; i < 6; i++) {
//             if (heuristicsConstrained[i][5]) {
//                 numberOfHeuristicConstraints++;
//             }
//             if (heuristicsConstrained[i][4]) {
//                 numberOfHeuristicObjectives++;
//             }
//         }
//
//         double dcThreshold = 0.5;
//         double massThreshold = 3000.0; // [kg]
//         double packEffThreshold = 0.7;
//         double instrCountThreshold = 15; // only for assigning problem
//         boolean considerFeasibility = true;
//
//         String savePath = System.getProperty("user.dir") + File.separator + "results";
//
//         //String resourcesPath = "C:\\Users\\dforn\\Documents\\TEXASAM\\PROJECTS\\VASSAR_resources"; // for laptop David
//         //String resourcesPath = "C:\\Users\\rosha\\Documents\\SEAK Lab Github\\VASSAR\\VASSAR_resources-heur"; // for laptop
// //         String resourcesPath = "C:\\Users\\dfornosf\\Documents\\VASSAR_resources";
//         String resourcesPath = "/home/ec2-user/vassar/VASSAR_resources";
//         this.l_params = new ClimateCentricAssigningParams(resourcesPath, "FUZZY-ATTRIBUTES", "test", "normal");
//
//         //PRNG.setRandom(new SynchronizedMersenneTwister());
//
//         HashMap<String, String[]> instrumentSynergyMap = getInstrumentSynergyNameMap(this.l_params);
//         HashMap<String, String[]> interferingInstrumentsMap = getInstrumentInterferenceNameMap(this.l_params);
//
//         ArchitectureEvaluator evaluator = new ArchitectureEvaluator(considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold);
//         ArchitectureEvaluationManager evaluationManager = new ArchitectureEvaluationManager(this.l_params, evaluator);
//         evaluationManager.init(numCpus);
//         this.l_problem = new AssigningProblem(new int[]{1}, this.l_params.getProblemName(), evaluationManager, evaluator, this.l_params, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold, instrCountThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);
//
//     }
//
//
//     public synchronized Future<Result> evaluateDesignAsync(ArrayList<Double> design){
//         AssigningArchitecture arch = new AssigningArchitecture(new int[]{1}, this.l_params.getNumInstr(), this.l_params.getNumOrbits(), 2);
//         StringBuilder bitStringBuilder = new StringBuilder(60);
//         for (int j = 1; j <= 60; ++j) {
//         //fort j = 1; j < arch.getNumberOfVariables(); ++j) {
//             BinaryVariable var = new BinaryVariable(1);
//             var.set(0, true);
//             if (design.get(j-1) == 0) {
//                 var.set(0, false);
//                 bitStringBuilder.append("0");
//             } else {
//                 var.set(0, true);
//                 bitStringBuilder.append("1");
//             }
//             arch.setVariable(j, var);
//         }
//         return this.l_problem.evaluateArchAsync(arch);
//     }
//
//
//
//     public double[] evaluateDesign(ArrayList<Double> design){
//
//         AssigningArchitecture arch = new AssigningArchitecture(new int[]{1}, this.l_params.getNumInstr(), this.l_params.getNumOrbits(), 2);
//         StringBuilder bitStringBuilder = new StringBuilder(60);
//
//         for (int j = 1; j <= 60; ++j) {
//         //fort j = 1; j < arch.getNumberOfVariables(); ++j) {
//             BinaryVariable var = new BinaryVariable(1);
//             var.set(0, true);
//             if (design.get(j-1) == 0) {
//                 var.set(0, false);
//                 bitStringBuilder.append("0");
//             } else {
//                 var.set(0, true);
//                 bitStringBuilder.append("1");
//             }
//             arch.setVariable(j, var);
//         }
//         this.l_problem.evaluateArch(arch);
//         return arch.getObjectives();
//     }
//
//
//
//
//
//
//
//
//
//
//
//
//     /**
//      * Creates instrument synergy map used to compute the instrument synergy violation heuristic (only formulated for the
//      * Climate Centric problem for now) (added by roshansuresh)
//      * @param params
//      * @return Instrument synergy hashmap
//      */
//     protected static HashMap<String, String[]> getInstrumentSynergyNameMap(BaseParams params) {
//         HashMap<String, String[]> synergyNameMap = new HashMap<>();
//         if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
//             synergyNameMap.put("ACE_ORCA", new String[]{"DESD_LID", "GACM_VIS", "ACE_POL", "HYSP_TIR", "ACE_LID"});
//             synergyNameMap.put("DESD_LID", new String[]{"ACE_ORCA", "ACE_LID", "ACE_POL"});
//             synergyNameMap.put("GACM_VIS", new String[]{"ACE_ORCA", "ACE_LID"});
//             synergyNameMap.put("HYSP_TIR", new String[]{"ACE_ORCA", "POSTEPS_IRS"});
//             synergyNameMap.put("ACE_POL", new String[]{"ACE_ORCA", "DESD_LID"});
//             synergyNameMap.put("ACE_LID", new String[]{"ACE_ORCA", "CNES_KaRIN", "DESD_LID", "GACM_VIS"});
//             synergyNameMap.put("POSTEPS_IRS", new String[]{"HYSP_TIR"});
//             synergyNameMap.put("CNES_KaRIN", new String[]{"ACE_LID"});
//         }
//         else {
//             System.out.println("Synergy Map for current problem not formulated");
//         }
//         return synergyNameMap;
//     }
//
//     /**
//      * Creates instrument interference map used to compute the instrument interference violation heuristic (only formulated for the
//      * Climate Centric problem for now)
//      * @param params
//      * @return Instrument interference hashmap
//      */
//     protected static HashMap<String, String[]> getInstrumentInterferenceNameMap(BaseParams params) {
//         HashMap<String, String[]> interferenceNameMap = new HashMap<>();
//         if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
//             interferenceNameMap.put("ACE_LID", new String[]{"ACE_CPR", "DESD_SAR", "CLAR_ERB", "GACM_SWIR"});
//             interferenceNameMap.put("ACE_CPR", new String[]{"ACE_LID", "DESD_SAR", "CNES_KaRIN", "CLAR_ERB", "ACE_POL", "ACE_ORCA", "GACM_SWIR"});
//             interferenceNameMap.put("DESD_SAR", new String[]{"ACE_LID", "ACE_CPR"});
//             interferenceNameMap.put("CLAR_ERB", new String[]{"ACE_LID", "ACE_CPR"});
//             interferenceNameMap.put("CNES_KaRIN", new String[]{"ACE_CPR"});
//             interferenceNameMap.put("ACE_POL", new String[]{"ACE_CPR"});
//             interferenceNameMap.put("ACE_ORCA", new String[]{"ACE_CPR"});
//             interferenceNameMap.put("GACM_SWIR", new String[]{"ACE_LID", "ACE_CPR"});
//         }
//         else {
//             System.out.println("Interference Map fpr current problem not formulated");
//         }
//         return interferenceNameMap;
//     }
//
//     public enum RunMode{
//         RandomPopulation,
//         EpsilonMOEA,
//     }
//
//     public enum InitializationMode{
//         InitializeRandom,
//         InitializationRandomAndInjected,
//     }
//
//
// }
//
//
//
