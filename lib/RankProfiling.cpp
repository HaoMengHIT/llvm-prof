#include "preheader.h"
#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <unordered_map>

#include "ValueUtils.h"
#include "ProfilingUtils.h"
#include "ProfileInstrumentations.h"
#include "ProfileDataTypes.h"

namespace {
	class RankProfiler : public llvm::ModulePass
	{
		public:
			static char ID;
			RankProfiler():ModulePass(ID) {};
			bool runOnModule(llvm::Module&) override;
	};
}

using namespace llvm;
using namespace lle;
char RankProfiler::ID = 0;
static RegisterPass<RankProfiler> X("insert-rank-profiling",
		"insert profiling for ranks of processes", false, false);

bool RankProfiler::runOnModule(llvm::Module &M)
{
	Function *Main = M.getFunction("main");
	if (Main == 0) {
		errs() << "WARNING: cannot insert edge profiling into a module"
			<< " with no main function!\n";
		return false;  // No main, no instrumentation!
	}

	CallInst* CommRank = NULL;
	Function* CommRankFunc = NULL;
	Instruction* Next = NULL;
	for(auto F = M.begin(), E = M.end(); F!=E; ++F){
		if (F->isDeclaration()) continue;
		for(auto I = inst_begin(*F), IE = inst_end(*F); I!=IE; ++I){
			CallInst* CI = dyn_cast<CallInst>(&*I);
			if(CI == NULL) continue;
			Value* CV = const_cast<CallInst*>(CI)->getCalledValue();
			Function* func = dyn_cast<Function>(lle::castoff(CV));
			if(func == NULL)
				errs()<<"No func!\n";
			StringRef str = func->getName();
			if(str.startswith("mpi_comm_rank_")){
				CommRank = CI;
				CommRankFunc = func;
				I++;
				Next = dyn_cast<Instruction>(&*I);
				I--;
				break;
			}
		}
	}

	IRBuilder<> Builder(M.getContext());
	Type* I32Ty = Type::getInt32Ty(M.getContext());
	Type*ATy = ArrayType::get(I32Ty, 1);
	GlobalVariable* Counters = new GlobalVariable(M, ATy, false,
			GlobalVariable::InternalLinkage, Constant::getNullValue(ATy),
			"RankCounters");

	Builder.SetInsertPoint(Next);
	// Create the getelementptr constant expression
	std::vector<Constant*> Indices(2);
	Indices[0] = Constant::getNullValue(Type::getInt32Ty(M.getContext()));
	Indices[1] = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
	Constant *ElementPtr =
		ConstantExpr::getGetElementPtr(Counters, Indices);
	Value* FortranDT = Builder.CreateLoad(CommRank->getArgOperand(1));
	Builder.CreateStore(FortranDT, ElementPtr);

	InsertProfilingInitCall(Main, "llvm_start_rank_profiling", Counters);
	return true;
}
