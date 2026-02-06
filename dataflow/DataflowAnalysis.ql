/**
 * @name Agent Dataflow Analysis
 * @description Analyze dataflow from model output to execution
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.ApiGraphs

/**
 * Source: model generated output
 * engine.generate() return value
 */
class ModelOutputSource extends DataFlow::Node {
  ModelOutputSource() {
    exists(Class cls, Function func, Call call |
      cls.getName() = "SeeActAgent" and
      func.getName() = "predict" and
      call.getFunc().(Attribute).getName() = "generate" and
      call.getFunc().(Attribute).getObject().(Attribute).getName() = "engine" and
      call.getFunc().(Attribute).getObject().(Attribute).getObject().(Name).getId() = "self" and
      call.getEnclosingModule() = cls.getEnclosingModule() and
      call.getScope().getScope*() = cls and
      call.getScope() = func and
      this.asExpr() = call
    )
  }
}

/**
 * Sink: action execution
 * perform_action() arguments
 */
class ExecutionSink extends DataFlow::Node {
  ExecutionSink() {
    exists(Class cls, Function func, Call call |
      cls.getName() = "SeeActAgent" and
      func.getName() = "execute" and
      call.getFunc().(Attribute).getName() = "perform_action" and
      call.getFunc().(Attribute).getObject().(Name).getId() = "self" and
      call.getEnclosingModule() = cls.getEnclosingModule() and
      call.getScope().getScope*() = cls and
      call.getScope() = func and
      this.asExpr() = call
    )
  }
}

/**
 * Configuration module
 */
module AgentFlowConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { 
    source instanceof ModelOutputSource 
  }

  predicate isSink(DataFlow::Node sink) { 
    sink instanceof ExecutionSink 
  }
}

/**
 * Inter Procedural Tracking
 */
class InterProceduralTracking extends TaintTracking::AdditionalTaintStep {
  override predicate step(DataFlow::Node nodeFrom, DataFlow::Node nodeTo) {
    exists(Call call, Function func |
      // The call invokes the function
      call.getFunc().(Name).getId() = func.getName() and
      // Taint flows from call argument to return value
      nodeFrom.asExpr() = call.getAnArg() and
      exists(DataFlow::Node funcInput, DataFlow::Node funcOutput |
        analyzeFunction(func, funcInput, funcOutput) and
        nodeTo.asExpr() = call
      )
    )
  }
}

/**
 * Track inter-procedural functions
 */
predicate analyzeFunction(Function func, DataFlow::Node input, DataFlow::Node output) {
  exists(Parameter param, Return ret |
    param = func.getAnArg() and
    ret.getScope() = func and
    // Input is the parameter
    input.asExpr().(Name).getId() = param.getName() and
    // Output is the return value
    output.asExpr() = ret.getValue()
  )
}

/**
 * find string operation
  */
predicate isStringProcessing(Expr expr, string description) {
  // Built-in string methods
  exists(Call call | expr = call |
    exists(string method |
      call.getFunc().(Attribute).getName() = method and
      method in ["strip", "lstrip", "rstrip", "replace", "split", "join", "upper", "lower", 
                 "capitalize", "title", "format", "removeprefix", "removesuffix"] and
      description = "String method: " + method
    )
  )
  or
  // re module operations
  exists(Call call | expr = call |
    exists(string func |
      call.getFunc().(Attribute).getObject().(Name).getId() = "re" and
      call.getFunc().(Attribute).getName() = func and
      func in ["search", "match", "findall", "finditer", "sub", "subn", "split", "compile"] and
      description = "Regex function: re." + func
    )
  )
  or
  // String slicing
  exists(Subscript subscript | expr = subscript |
    subscript.getIndex() instanceof Slice and
    description = "String slicing: [" + subscript.getIndex().toString() + "]"
  )
  or
  // String formatting (f-strings, %, format)
  exists(FormattedValue fv | expr = fv and description = "f-string formatting")
  or
  exists(BinaryExpr binop | expr = binop |
    binop.getOp() instanceof Mod and description = "String formatting: %"
  )
//   or
//   // Pattern methods
//   exists(Call call | expr = call |
//     exists(string method |
//       call.getFunc().(Attribute).getName() = method and
//       method in ["search", "match", "findall", "finditer", "sub", "subn", "split"] and
//       description = "Pattern method: " + method
//     )
//   )
}


/**
 * Global taint tracking instance
 */
module AgentFlow = TaintTracking::Global<AgentFlowConfig>;
import AgentFlow::PathGraph

// query filtered string peocessing
// from AgentFlow::PathNode source, AgentFlow::PathNode sink, 
//      AgentFlow::PathNode mid, string description
// where 
//   AgentFlow::flowPath(source, sink) and
//   AgentFlow::flowPath(source, mid) and
//   AgentFlow::flowPath(mid, sink) and
//   isStringProcessing(mid.getNode().asExpr(), description)
// select mid, source, sink, description

// basic query
// from AgentFlow::PathNode source, AgentFlow::PathNode sink
// where AgentFlow::flowPath(source, sink)
// select sink.getNode(), source, sink, "Flow from model output to execution"

from AgentFlow::PathNode source, AgentFlow::PathNode sink, 
     AgentFlow::PathNode mid
where 
  AgentFlow::flowPath(source, sink) and
  mid = source.getASuccessor*() and
  sink = mid.getASuccessor*() and
  mid != source and mid != sink  // exclude source and sink themselves
select mid, mid.getNode().asExpr(), mid.getNode().toString()
