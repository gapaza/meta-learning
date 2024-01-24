/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package rbs.transformer;
import py4j.GatewayServer;
// import rbs.transformer.EvaluationAssigning;
// import rbs.transformer.EvaluationGigaAssigning;
import rbs.transformer.EvaluationTruss;
// import rbs.transformer.EvaluationAssigningParallel;


public class App {
    public String getGreeting() {
        return "Hello World!";
    }

    public static void main(String[] args) {
//         EvaluationAssigning myJavaObject = new EvaluationAssigning();
//         EvaluationAssigningParallel myJavaObject = new EvaluationAssigningParallel();
//         EvaluationGigaAssigning myJavaObject = new EvaluationGigaAssigning();
//         EvaluationPartitioning myJavaObject = new EvaluationPartitioning();
        EvaluationTruss myJavaObject = new EvaluationTruss();
        py4j.GatewayServer gatewayServer = new py4j.GatewayServer(myJavaObject);
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}
