#define DEBUG_TYPE "insert-edge-profiling"
#include "preheader.h"
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/IR/IRBuilder.h>
#include "ProfilingUtils.h"
#include "InitializeProfilerPass.h"
#include "ProfileInstrumentations.h"
#include <set>
#include "ValueUtils.h"
#include "ProfilingUtils.h"
#include "ProfileDataTypes.h"
using namespace llvm;
using namespace lle;

STATISTIC(NumEdgesInserted, "The # of edges inserted.");

namespace {
   class EdgeRankProfiler : public ModulePass {
      bool runOnModule(Module &M);
      public:
      static char ID; // Pass identification, replacement for typeid
      EdgeRankProfiler() : ModulePass(ID) {
         //initializeEdgeProfilerPass(*PassRegistry::getPassRegistry());
      }

      virtual const char *getPassName() const {
         return "Edge Rank Profiler";
      }
   };
}

char EdgeRankProfiler::ID = 0;
INITIALIZE_PASS(EdgeRankProfiler, "insert-edge-rank-profiling",
      "Insert instrumentation for edge profiling for rank", false, false)

// regist pass for opt load
static RegisterPass<EdgeRankProfiler> X("insert-edge-rank-profiling",
      "Insert instrumentation for edge profiling for rank",false,false);

ModulePass *llvm::createEdgeRankProfilerPass() { return new EdgeRankProfiler(); }

bool EdgeRankProfiler::runOnModule(Module &M) {
   Function *Main = M.getFunction("main");
   if (Main == 0) {
      errs() << "WARNING: cannot insert edge profiling into a module"
         << " with no main function!\n";
      return false;  // No main, no instrumentation!
   }

   std::set<BasicBlock*> BlocksToInstrument;
   unsigned NumEdges = 0;
   for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      // Reserve space for (0,entry) edge.
      ++NumEdges;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
         // Keep track of which blocks need to be instrumented.  We don't want to
         // instrument blocks that are added as the result of breaking critical
         // edges!
         BlocksToInstrument.insert(BB);
         NumEdges += BB->getTerminator()->getNumSuccessors();
      }
   }

   Type *ATy = ArrayType::get(Type::getInt64Ty(M.getContext()), NumEdges);
   GlobalVariable *Counters =
      new GlobalVariable(M, ATy, false, GlobalValue::InternalLinkage,
            Constant::getNullValue(ATy), "EdgeProfCounters");
   NumEdgesInserted = NumEdges;

   // Instrument all of the edges...
   unsigned i = 0;
   for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
      if (F->isDeclaration()) continue;
      // Create counter for (0,entry) edge.
      IncrementCounterInBlock(&F->getEntryBlock(), i++, Counters);
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
         if (BlocksToInstrument.count(BB)) {  // Don't instrument inserted blocks
            // Okay, we have to add a counter of each outgoing edge.  If the
            // outgoing edge is not critical don't split it, just insert the counter
            // in the source or destination of the edge.
            TerminatorInst *TI = BB->getTerminator();
            for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
               // If the edge is critical, split it.
               SplitCriticalEdge(TI, s, this);

               // Okay, we are guaranteed that the edge is no longer critical.  If we
               // only have a single successor, insert the counter in this block,
               // otherwise insert it in the successor block.
               if (TI->getNumSuccessors() == 1) {
                  // Insert counter at the end of the block
                  IncrementCounterInBlock(BB, i++, Counters, false);
               } else {
                  // Insert counter at the start of the block
                  IncrementCounterInBlock(TI->getSuccessor(s), i++, Counters);
               }
            }
         }
   }
	CallInst* CommRank = NULL;
	Function* CommRankFunc = NULL;
	Instruction* Next = NULL;
	for(auto F = M.begin(), E = M.end(); F!=E; ++F){
		for(auto I = inst_begin(*F), IE = inst_end(*F); I!=IE; ++I){
			CallInst* CI = dyn_cast<CallInst>(&*I);
			if(CI == NULL) continue;
			Value* CV = const_cast<CallInst*>(CI)->getCalledValue();
			Function* func = dyn_cast<Function>(castoff(CV));
			if(func == NULL)
			{
				errs()<<"No func!\n";
				continue;
			}
			StringRef str = func->getName();
			if(str.startswith("mpi_comm_rank_")||str.startswith("MPI_Comm_rank")){
				CommRank = CI;
				CommRankFunc = func;
				++I;
				Next = dyn_cast<Instruction>(&*I);
            break;
			}
		}
	}
	IRBuilder<> Builder(M.getContext());
   Type* I32Ty = Type::getInt32Ty(M.getContext());
   Type*ATyI32 = ArrayType::get(I32Ty, 1);
   GlobalVariable* RankCounters = new GlobalVariable(M, ATyI32, false,
         GlobalVariable::InternalLinkage, Constant::getNullValue(ATyI32),
         "RankCounters");

	Builder.SetInsertPoint(Next);
   Value* RankLoad = Builder.CreateLoad(CommRank->getArgOperand(1));
   // Create the getelementptr constant expression
   std::vector<Constant*> Indices(2);
   Indices[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
   Indices[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
   Constant *ElementPtr =
      ConstantExpr::getGetElementPtr(RankCounters, Indices);
   Builder.CreateStore(RankLoad, ElementPtr);

   // Add the initialization call to main.
   // InsertPredMPIProfilingInitCall
   InsertPredMPIProfilingInitCall(Main, "llvm_start_edge_rank_profiling", Counters, RankCounters);
   return true;
}

