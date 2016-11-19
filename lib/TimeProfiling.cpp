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
	class TimeProfiler : public llvm::ModulePass
	{
		public:
			static char ID;
			TimeProfiler():ModulePass(ID) {};
			bool runOnModule(llvm::Module&) override;
	};
}

using namespace llvm;
using namespace lle;
char TimeProfiler::ID = 0;
static RegisterPass<TimeProfiler> X("insert-time-profiling",
		"insert profiling for the time of mpi communication", false, false);

//Get the next instruction after Point
static Value* getNextIns(Value* Point)
{
	BasicBlock* pb = dyn_cast<Instruction>(Point)->getParent();
	for (BasicBlock::iterator i = pb->begin(), e = pb->end(); i != e; ++i)
	{
		if((&*i) == Point)
		{
			BasicBlock::iterator ii = ++i;
			return &*ii;
		}
	}
}
static void IncrementTimeCounter(Value* Inc, Value* calle, unsigned Index, GlobalVariable* Counters, IRBuilder<>& Builder, Value* Point)
{
	LLVMContext &Context = Inc->getContext();
	//In order to insert instruction after Point, we use the nextIns function.
	Value* nextIns = getNextIns(Point);
	Builder.SetInsertPoint(dyn_cast<Instruction>(nextIns));

	// Create the getelementptr constant expression
	std::vector<Constant*> Indices(2);
	Indices[0] = Constant::getNullValue(Type::getInt32Ty(Context));
	Indices[1] = ConstantInt::get(Type::getInt32Ty(Context), Index);
	Constant *ElementPtr =
		ConstantExpr::getGetElementPtr(Counters, Indices);

	// Load, increment and store the value back.
	// Use this formula: a = a + end_time - start_time
	ArrayRef<Value*> args;
	CallInst* Inc_end = Builder.CreateCall(calle, args, "");
	Value* OldVal = Builder.CreateLoad(ElementPtr, "OldTimeCounter");
	Value* TmpVal = Builder.CreateFSub(OldVal, Inc, "TmpTimeCounter");
	Value* NewVal = Builder.CreateFAdd(TmpVal, Inc_end, "NewTimeCounter");
	Builder.CreateStore(NewVal, ElementPtr);
}

bool TimeProfiler::runOnModule(llvm::Module &M)
{
	Function *Main = M.getFunction("main");
	if (Main == 0) {
		errs() << "WARNING: cannot insert edge profiling into a module"
			<< " with no main function!\n";
		return false;  // No main, no instrumentation!
	}

	std::vector<CallInst*> Traped;
	Function* wtime = NULL;
	CallInst* CommRank = NULL;
	Function* CommRankFunc = NULL;
	Instruction* Next = NULL;
	for(auto F = M.begin(), E = M.end(); F!=E; ++F){
		if((*F).getName() == "mpi_wtime_")
			wtime = &*F;
		for(auto I = inst_begin(*F), IE = inst_end(*F); I!=IE; ++I){
			CallInst* CI = dyn_cast<CallInst>(&*I);
			if(CI == NULL) continue;
			Value* CV = const_cast<CallInst*>(CI)->getCalledValue();
			Function* func = dyn_cast<Function>(castoff(CV));
			if(func == NULL)
				errs()<<"No func!\n";
			StringRef str = func->getName();
			if(str.startswith("mpi_comm_rank_")){
				CommRank = CI;
				CommRankFunc = func;
				++I;
				Next = dyn_cast<Instruction>(&*I);
				--I;
			}
			if(str.startswith("mpi_"))
			{
				if(str.startswith("mpi_init_")||str.startswith("mpi_comm_rank_")||str.startswith("mpi_comm_size_"))
					continue;
				Traped.push_back(CI);
			}
		}
	}
	if(wtime == NULL)
	{
		std::vector<Type*> Doubles;
		FunctionType *Ft = FunctionType::get(Type::getDoubleTy(Main->getContext()),Doubles,false);
		wtime  = Function::Create(Ft, Function::ExternalLinkage,"mpi_wtime_",&M);
	}

	IRBuilder<> Builder(M.getContext());

	Type* DoubleTy = Type::getDoubleTy(M.getContext());

	Type*ATy = ArrayType::get(DoubleTy, Traped.size());
	GlobalVariable* Counters = new GlobalVariable(M, ATy, false,
			GlobalVariable::InternalLinkage, Constant::getNullValue(ATy),
			"TimeCounters");

	unsigned I=0;
	for(auto P : Traped){
		ArrayRef<Value*> args;
		CallInst* callTime = CallInst::Create(wtime, args, "", P);

		IncrementTimeCounter(callTime, wtime, I++, Counters, Builder, P);
	}

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

	InsertPredMPIProfilingInitCall(Main, "llvm_start_time_profiling", Counters ,RankCounters);
	return true;
}
