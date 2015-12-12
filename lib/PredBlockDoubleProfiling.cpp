#include "preheader.h"
#include "PredBlockDoubleProfiling.h"
#include "ProfilingUtils.h"

#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
using namespace std;

char PredBlockDoubleProfiler::ID = 0;
PredBlockDoubleProfiler* PredBlockDoubleProfiler::ins = NULL;

static RegisterPass<PredBlockDoubleProfiler> X("insert-pred-double-profiling", "Insert Block(double type of array) Predicate Profiling into Module", false, true);

PredBlockDoubleProfiler::PredBlockDoubleProfiler():ModulePass(ID)
{
   PredBlockDoubleProfiler::ins = this;
}


static void IncrementBlockCounters(llvm::Value* Inc, unsigned Index, GlobalVariable* Counters, IRBuilder<>& Builder)
{
   LLVMContext &Context = Inc->getContext();

   // Create the getelementptr constant expression
   std::vector<Constant*> Indices(2);
   Indices[0] = Constant::getNullValue(Type::getInt32Ty(Context));
   Indices[1] = ConstantInt::get(Type::getInt32Ty(Context), Index);
   Constant *ElementPtr =
      ConstantExpr::getGetElementPtr(Counters, Indices);

   // Load, increment and store the value back.
   Value* OldVal = Builder.CreateLoad(ElementPtr, "OldBlockCounter");
   Value* NewVal = Builder.CreateFAdd(
       OldVal, Builder.CreateSIToFP(Inc, Type::getDoubleTy(Context)),
       "NewBlockCounter");
   Builder.CreateStore(NewVal, ElementPtr);
}

bool PredBlockDoubleProfiler::runOnModule(Module& M)
{
   unsigned Idx = 0;
   IRBuilder<> Builder(M.getContext());

   unsigned NumBlocks = 0;
   for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
      NumBlocks += F->size();

	Type*ATy = ArrayType::get(Type::getDoubleTy(M.getContext()),NumBlocks);
	GlobalVariable* Counters = new GlobalVariable(M, ATy, false,
			GlobalVariable::InternalLinkage, Constant::getNullValue(ATy),
			"BlockPredCounters");

   for(auto F = M.begin(), FE = M.end(); F != FE; ++F){
      for(auto BB = F->begin(), BBE = F->end(); BB != BBE; ++BB){
         auto Found = BlockTraps.find(BB);
         if(Found == BlockTraps.end()){
            Idx++;
         }else{
            Value* Inc = Found->second.first;
            Value* InsertPos = Found->second.second;
            if(BasicBlock* InsertBB = dyn_cast<BasicBlock>(InsertPos))
               Builder.SetInsertPoint(InsertBB);
            else if(Instruction* InsertI = dyn_cast<Instruction>(InsertPos))
               Builder.SetInsertPoint(InsertI);
            else assert(0 && "unknow insert position type");
            IncrementBlockCounters(Inc, Idx++, Counters, Builder);
         }
      }
   }

	Function* Main = M.getFunction("main");
	InsertPredProfilingInitCall(Main, "llvm_start_pred_double_block_profiling", Counters);
	return true;
}
