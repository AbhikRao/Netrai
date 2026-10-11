function tests=testAggregateValidationRuns
% Synthetic evidence-contract checks; no model or patient image is loaded.
    tests=functiontests(localfunctions);
end

function setupOnce(testCase)
    base=fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath=path;
    addpath(base,genpath(fullfile(base,'src')));
end

function teardownOnce(testCase)
    path(testCase.TestData.originalPath);
end

function setup(testCase)
    testCase.TestData.root=tempname;
    mkdir(testCase.TestData.root);
end

function teardown(testCase)
    if isfolder(testCase.TestData.root), rmdir(testCase.TestData.root,'s'); end
end

function testCompleteLegacyCohort(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    expected=fullfile(root,'expected.csv');
    writetable(table(rows.id_code,rows.true_grade,'VariableNames',{'id_code','diagnosis'}),expected);
    identity.selection_sha256=fileSHA256(expected);
    writeChunk(root,'a',rows(1,:),identity); writeChunk(root,'b',rows(2,:),identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'),'ExpectedSelectionCsv',expected);
    verifyEqual(testCase,result.successful,2);
    verifyEqual(testCase,result.referable_auc,1);
    verifyTrue(testCase,result.expected_cohort_verified);
    verifyFalse(testCase,result.decision_policy_verified);
    verifyEqual(testCase,result.coverage_scope,"expected_cohort");
end

function testLegacyWithoutExpectedCohortIsExplicitlyPartial(testCase)
    root=testCase.TestData.root;
    writeChunk(root,'a',validRows,validIdentity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyFalse(testCase,result.expected_cohort_verified);
    verifyFalse(testCase,result.decision_policy_verified);
    verifyEqual(testCase,result.coverage_scope,"observed_chunks");
end

function testMissingAndMalformedIdentitiesFail(testCase)
    root=testCase.TestData.root; rows=validRows;
    writeChunk(root,'a',rows,struct);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkIdentity');
    identity=validIdentity; identity.model_sha256='bogus';
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkIdentity');
    delete(fullfile(root,'a','run_identity.json'));
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:UnboundChunk');
end

function testMixedModelsAndDuplicateCasesFail(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    writeChunk(root,'a',rows(1,:),identity);
    changed=identity; changed.config_sha256=repmat('c',1,64);
    writeChunk(root,'b',rows(2,:),changed);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:IncompatibleChunks');
    conflict=rows(1,:); conflict.true_grade=4;
    writeChunk(root,'b',conflict,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:DuplicateChunkIDs');
end

function testInvalidScoresAndBooleanNaNsFail(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    for bad=[NaN,Inf,-0.1,1.1]
        changed=rows; changed.referable_probability(1)=bad;
        writeChunk(root,'a',changed,identity);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkProbability');
        changed=rows; changed.referable_head_probability(1)=bad;
        writeChunk(root,'a',changed,identity);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkProbability');
    end
    changed=rows; changed.head_disagreement(1)=NaN;
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkBoolean');
    changed=rows; changed.referable_head_prediction(1)=2;
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkBoolean');
    verifyFalse(testCase,isfolder(fullfile(root,'combined')));
end

function testPendingRowsInvalidTruthAndDecisionsFail(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    changed=rows; changed.status(1)="pending";
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkStatus');
    changed=rows; changed.status(1)="error"; changed.true_grade(1)=NaN;
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidGrade');
    changed=rows; changed.head_disagreement(1)=1;
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkDecision');
    changed=rows; changed.triage_action(1)="refer";
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkDecision');
end

function testMalformedGradeProbabilityVectorsFail(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    for index=0:4, rows.(sprintf('prob_grade_%d',index))=[0.2;0.2]; end
    rows.prob_grade_0(1)=0.1;
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkProbability');
    rows.prob_grade_0(1)=0.2; rows.prob_grade_4=[];
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkProbability');
end

function testConsistentRawAndCalibratedFieldsAreAccepted(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    writeChunk(root,'a',rows,validIdentity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.successful,2);
    verifyEqual(testCase,result.referable_auc,1);
    % Grade 2 uses its calibrated probability, even though Grade 3 is larger.
    verifyEqual(testCase,rows.confidence(2),rows.prob_grade_2(2));
    verifyGreaterThan(testCase,rows.prob_grade_3(2),rows.confidence(2));
end

function testRawReferralScalarMustMatchItsVector(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.referable_probability=flipud(rows.referable_probability);
    writeChunk(root,'a',rows,validIdentity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkResult');
    verifyFalse(testCase,isfolder(fullfile(root,'combined')));
end

function testCalibratedReferralScalarMustMatchItsVector(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.calibrated_referable_probability(1)=rows.referable_probability(2);
    writeChunk(root,'a',rows,validIdentity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkResult');
end

function testContinuousScoreMustUseRawProbabilities(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.continuous_score(1)=rows.continuous_score(1)+0.1;
    writeChunk(root,'a',rows,validIdentity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkResult');
end

function testConfidenceMustUseThePredictedGrade(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.confidence(2)=rows.prob_grade_3(2);
    writeChunk(root,'a',rows,validIdentity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkResult');
end

function testOnlySerializationScaleDifferencesAreAccepted(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.referable_probability(1)=rows.referable_probability(1)+5e-13;
    writeChunk(root,'a',rows,validIdentity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.successful,2);
    rmdir(fullfile(root,'merged'),'s');
    rows.referable_probability(1)=rows.referable_probability(1)+1e-8;
    writeChunk(root,'a',rows,validIdentity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkResult');
end

function testUnresolvedVectorsDoNotRequireDerivedPredictions(testCase)
    root=testCase.TestData.root; rows=consistentProbabilityRows;
    rows.status=["error";"recapture"]; rows.predicted_grade(:)=NaN;
    for name=["referable_probability","calibrated_referable_probability", ...
            "confidence","continuous_score","referable_head_probability", ...
            "prob_grade_"+string(0:4),"raw_prob_grade_"+string(0:4)]
        rows.(name)(:)=NaN;
    end
    writeChunk(root,'a',rows,validIdentity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.successful,0);
    verifyEqual(testCase,result.unresolved,2);
end

function testMissingExtraAndRelabelledExpectedCasesFail(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    expected=fullfile(root,'expected.csv');
    writetable(table(rows.id_code,rows.true_grade,'VariableNames',{'id_code','diagnosis'}),expected);
    writeChunk(root,'a',rows(1,:),identity);
    verifyError(testCase,@() aggregateValidationRuns(root,fullfile(root,'merged'), ...
        'ExpectedCohortCsv',expected),'NetrAI:IncompleteCohort');
    changed=rows; changed.true_grade(1)=1;
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root,fullfile(root,'merged'), ...
        'ExpectedCohortCsv',expected),'NetrAI:ConflictingCohortLabels');
    changed=rows; changed.id_code(2)="extra";
    writeChunk(root,'a',changed,identity);
    verifyError(testCase,@() aggregateValidationRuns(root,fullfile(root,'merged'), ...
        'ExpectedCohortCsv',expected),'NetrAI:IncompleteCohort');
end

function testExpectedSelectionDigestMustMatch(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    expected=fullfile(root,'expected.csv');
    writetable(table(rows.id_code,rows.true_grade,'VariableNames',{'id_code','diagnosis'}),expected);
    identity.selection_sha256=repmat('b',1,64);
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root,fullfile(root,'merged'), ...
        'ExpectedSelectionCsv',expected),'NetrAI:IncompatibleExpectedCohort');
end

function testErrorsAndRecapturesRemainUnresolved(testCase)
    root=testCase.TestData.root; rows=validRows;
    failed=rows(1,:); failed.id_code="failed"; failed.status="error";
    failed.predicted_grade=NaN; failed.referable_probability=NaN;
    recapture=failed; recapture.id_code="recapture"; recapture.status="recapture";
    writeChunk(root,'a',[rows;failed;recapture],validIdentity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.requested,4); verifyEqual(testCase,result.successful,2);
    verifyEqual(testCase,result.errors,1); verifyEqual(testCase,result.recaptures,1);
    verifyEqual(testCase,result.unresolved,2); verifyEqual(testCase,result.coverage,0.5);
end

function testVersionTwoBindsCompleteCasesAndImageHashes(testCase)
    root=testCase.TestData.root; rows=validRows; identity=versionTwo(rows);
    rows.image_sha256=repmat(string(repmat('d',1,64)),2,1);
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.expected_cohort_verified);
    verifyFalse(testCase,result.decision_policy_verified);
    rmdir(fullfile(root,'merged'),'s');
    writeChunk(root,'a',rows(1,:),identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:IncompleteCohort');
    rows.image_sha256(1)=string(repmat('e',1,64));
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:ConflictingImageIdentity');
end

function testVersionTwoMissingImageIsUnresolved(testCase)
    root=testCase.TestData.root; rows=validRows; identity=versionTwo(rows);
    identity.expected_cohort(1).image_sha256='';
    rows.image_sha256=["";string(repmat('d',1,64))];
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:UnboundSuccessfulImage');
    rows.status(1)="error"; rows.predicted_grade(1)=NaN;
    rows.image_sha256=["";string(repmat('d',1,64))];
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.expected_cohort_verified);
    verifyEqual(testCase,result.coverage,0.5);
end

function testNumericLookingIdsKeepLeadingZeros(testCase)
    root=testCase.TestData.root; rows=validRows;
    rows.id_code=["001";"002"];
    rows.image_sha256=repmat(string(repmat('d',1,64)),2,1);
    identity=versionTwo(rows);
    writeChunk(root,'a',rows,identity);
    aggregateValidationRuns(root,fullfile(root,'merged'));
    contents=fileread(fullfile(root,'merged','per_image_results.csv'));
    verifyTrue(testCase,contains(contents,'001,'));
    verifyTrue(testCase,contains(contents,'002,'));
end

function testAllErroredCohortPublishesUnresolvedAccounting(testCase)
    root=testCase.TestData.root; rows=validRows; identity=versionTwo(rows);
    rows.status=["error";"recapture"];
    rows.predicted_grade(:)=NaN; rows.referable_probability(:)=NaN;
    rows.referable_head_probability(:)=NaN;
    rows.image_sha256=repmat(string(repmat('d',1,64)),2,1);
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.requested,2); verifyEqual(testCase,result.successful,0);
    verifyEqual(testCase,result.errors,1); verifyEqual(testCase,result.recaptures,1);
    verifyEqual(testCase,result.unresolved,2); verifyEqual(testCase,result.coverage,0);
    verifyTrue(testCase,isnan(result.qwk));
    verifyTrue(testCase,isnan(result.referable_sensitivity));
    verifyTrue(testCase,isfile(fullfile(root,'merged','per_image_results.csv')));
    text=fileread(fullfile(root,'merged','aggregate_metrics.json'));
    verifyTrue(testCase,contains(text,'"qwk": null'));
end

function testExistingResultsAreNeverOverwritten(testCase)
    root=testCase.TestData.root; rows=validRows;
    writeChunk(root,'a',rows,validIdentity);
    destination=fullfile(root,'merged'); mkdir(destination);
    sentinel=fullfile(destination,'keep.txt');
    fid=fopen(sentinel,'w'); fprintf(fid,'original'); fclose(fid);
    verifyError(testCase,@() aggregateValidationRuns(root,destination),'NetrAI:OutputExists');
    verifyEqual(testCase,fileread(sentinel),'original');
    delete(sentinel);
    result=aggregateValidationRuns(root,destination);
    verifyEqual(testCase,result.successful,2);
    prior=fileread(fullfile(destination,'aggregate_metrics.json'));
    verifyError(testCase,@() aggregateValidationRuns(root,destination),'NetrAI:OutputExists');
    verifyEqual(testCase,fileread(fullfile(destination,'aggregate_metrics.json')),prior);
end

function testVersionTwoRequiresRowImageDigests(testCase)
    root=testCase.TestData.root; rows=validRows;
    writeChunk(root,'a',rows,versionTwo(rows));
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:UnboundChunk');
end

function testUnpublishedStagingIsExcluded(testCase)
    root=testCase.TestData.root; rows=validRows; identity=validIdentity;
    writeChunk(root,'a',rows,identity);
    writeChunk(root,'.netrai-staging-incomplete',rows,struct);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyEqual(testCase,result.requested,2);
    verifyEqual(testCase,result.chunks_merged,1);
end

function testVersionThreeVerifiesContentBoundDecisionPolicy(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.expected_cohort_verified);
    verifyTrue(testCase,result.decision_policy_verified);
    verifyEqual(testCase,result.successful,2);
end

function testVersionThreeRequiresPolicyAndConfigSnapshot(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    for field=["decision_policy","config_json"]
        changed=rmfield(identity,field);
        writeChunk(root,'a',rows,changed);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkPolicy');
    end
end

function testVersionThreeRejectsMalformedPolicy(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    invalid={ ...
        struct('grade_thresholds',[0.5 1.5 1.5 3.5]), ...
        struct('grade_thresholds',[0 1.5 2.5 3.5]), ...
        struct('grade_thresholds',[0.5 1.5 2.5 4]), ...
        struct('grade_thresholds',[0.5 1.5 NaN 3.5]), ...
        struct('grade_thresholds','0.5,1.5,2.5,3.5'), ...
        struct('referable_head_threshold',0), ...
        struct('referable_head_threshold',1), ...
        struct('referable_head_threshold',[0.5 0.6]), ...
        struct('referable_head_threshold','0.5'), ...
        struct('confidence_temperature',0), ...
        struct('confidence_temperature',Inf), ...
        struct('confidence_temperature',[0.8 1]), ...
        struct('unsupported',true)};
    for index=1:numel(invalid)
        changed=identity; fields=fieldnames(invalid{index});
        changed.decision_policy.(fields{1})=invalid{index}.(fields{1});
        writeChunk(root,'a',rows,changed);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkPolicy');
    end
end

function testVersionThreePolicyValuesMustMatchFrozenConfig(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    values=struct('grade_thresholds',[0.6 1.5 2.5 3.5], ...
        'referable_head_threshold',0.6,'confidence_temperature',0.9);
    for field=string(fieldnames(values))'
        changed=identity; changed.decision_policy.(field)=values.(field);
        writeChunk(root,'a',rows,changed);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:ConflictingDecisionPolicy');
    end
end

function testVersionThreeConfigJsonMustMatchHash(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    changed=identity; changed.config_json=[changed.config_json ' '];
    writeChunk(root,'a',rows,changed);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkPolicy');
    changed.config_json='{invalid json';
    changed.config_sha256=sha256Bytes(unicode2native(changed.config_json,'UTF-8'));
    writeChunk(root,'a',rows,changed);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkPolicy');
end

function testVersionThreeRejectsForgedGradeWithConsistentOtherFields(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    rows.predicted_grade(1)=4;
    rows.confidence(1)=rows.prob_grade_4(1);
    rows.head_disagreement(1)=1;
    rows.triage_action(1)="uncertain_human_review";
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkDecision');
    verifyFalse(testCase,isfolder(fullfile(root,'combined')));
end

function testVersionThreeRejectsForgedReferralHeadWithConsistentRouting(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    rows.referable_head_prediction(1)=1;
    rows.head_disagreement(1)=1;
    rows.triage_action(1)="uncertain_human_review";
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkDecision');
end

function testVersionThreeRejectsWrongCalibrationWithConsistentDerivedFields(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    for grade=0:4
        rows.(sprintf('prob_grade_%d',grade))=rows.(sprintf('raw_prob_grade_%d',grade));
    end
    rows.calibrated_referable_probability=rows.referable_probability;
    rows.confidence=[rows.prob_grade_0(1);rows.prob_grade_2(2)];
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkCalibration');
end

function testVersionThreeRequiresFullProbabilityVectors(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    for field=["raw_prob_grade_4","prob_grade_4","continuous_score","triage_action"]
        changed=removevars(rows,field);
        writeChunk(root,'a',changed,identity);
        verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InvalidChunkPolicy');
    end
end

function testVersionThreeRejectsDifferentFrozenPoliciesAcrossChunks(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    writeChunk(root,'a',rows(1,:),identity);
    changed=identity;
    changed.decision_policy.confidence_temperature=0.9;
    changed.config_json=jsonencode(changed.decision_policy);
    changed.config_sha256=sha256Bytes(unicode2native(changed.config_json,'UTF-8'));
    rows=calibratedRows(rows,changed.decision_policy.confidence_temperature);
    writeChunk(root,'b',rows(2,:),changed);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:IncompatibleChunks');
end

function testVersionThreeUsesItsOwnFrozenThresholdAndTemperature(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    identity.decision_policy.grade_thresholds=[0.2 0.8 2.5 3.8];
    identity.decision_policy.referable_head_threshold=0.95;
    identity.decision_policy.confidence_temperature=2;
    identity.config_json=jsonencode(identity.decision_policy);
    identity.config_sha256=sha256Bytes(unicode2native(identity.config_json,'UTF-8'));
    rows.predicted_grade=[1;2]; rows.referable_head_prediction=[0;0];
    rows.head_disagreement=[0;1]; rows.triage_action=["auto_clear";"uncertain_human_review"];
    rows=calibratedRows(rows,2);
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.decision_policy_verified);
    verifyEqual(testCase,result.head_disagreement_count,1);
end

function testVersionThreeIncludesEqualityInReferralThreshold(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    rows.referable_head_probability(1)=identity.decision_policy.referable_head_threshold;
    rows.referable_head_prediction(1)=1;
    rows.head_disagreement(1)=1; rows.triage_action(1)="uncertain_human_review";
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.decision_policy_verified);
    rmdir(fullfile(root,'merged'),'s');
    rows.referable_head_prediction(1)=0;
    rows.head_disagreement(1)=0; rows.triage_action(1)="auto_clear";
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkDecision');
end

function testVersionThreeIncludesEqualityInOrdinalThreshold(testCase)
    root=testCase.TestData.root; [rows,identity]=policyRows;
    for grade=0:4, rows.(sprintf('raw_prob_grade_%d',grade))(1)=0; end
    rows.raw_prob_grade_0(1)=0.5; rows.raw_prob_grade_1(1)=0.5;
    rows.continuous_score(1)=0.5; rows.referable_probability(1)=0;
    rows.predicted_grade(1)=1;
    rows=calibratedRows(rows,identity.decision_policy.confidence_temperature);
    writeChunk(root,'a',rows,identity);
    result=aggregateValidationRuns(root,fullfile(root,'merged'));
    verifyTrue(testCase,result.decision_policy_verified);
    rmdir(fullfile(root,'merged'),'s');
    rows.predicted_grade(1)=0; rows.confidence(1)=rows.prob_grade_0(1);
    writeChunk(root,'a',rows,identity);
    verifyError(testCase,@() aggregateValidationRuns(root),'NetrAI:InconsistentChunkDecision');
end

function [rows,identity]=policyRows
    rows=consistentProbabilityRows;
    rows.image_sha256=repmat(string(repmat('d',1,64)),height(rows),1);
    identity=versionTwo(rows); identity.schema_version=3;
    identity.decision_policy=struct('grade_thresholds',[0.5 1.5 2.5 3.5], ...
        'referable_head_threshold',0.5,'confidence_temperature',0.8);
    identity.config_json=jsonencode(identity.decision_policy);
    identity.config_sha256=sha256Bytes(unicode2native(identity.config_json,'UTF-8'));
    rows=calibratedRows(rows,identity.decision_policy.confidence_temperature);
end

function rows=calibratedRows(rows,temperature)
    raw=rows{:,cellstr("raw_prob_grade_"+string(0:4))};
    calibrated=zeros(size(raw));
    for row=1:height(rows)
        calibrated(row,:)=calibrateConfidence(raw(row,:),temperature);
    end
    for grade=0:4, rows.(sprintf('prob_grade_%d',grade))=calibrated(:,grade+1); end
    rows.calibrated_referable_probability=sum(calibrated(:,3:5),2);
    selected=sub2ind(size(calibrated),(1:height(rows))',rows.predicted_grade+1);
    rows.confidence=calibrated(selected);
end

function rows=validRows
    rows=table(["a";"b"],[0;2],[0;2],[0.1;0.9],[0.1;0.9],[0;1],[0;0], ...
        ["success";"success"],["auto_clear";"refer"],'VariableNames', ...
        {'id_code','true_grade','predicted_grade','referable_probability', ...
        'referable_head_probability','referable_head_prediction','head_disagreement','status','triage_action'});
end

function rows=consistentProbabilityRows
    rows=validRows;
    raw=[0.8 0.1 0.04 0.03 0.03; 0.15 0.01 0.39 0.4 0.05];
    calibrated=[0.85 0.05 0.05 0.03 0.02; 0.11 0.004 0.412 0.425 0.049];
    for index=0:4
        rows.(sprintf('raw_prob_grade_%d',index))=raw(:,index+1);
        rows.(sprintf('prob_grade_%d',index))=calibrated(:,index+1);
    end
    rows.referable_probability=[0.1;0.84];
    rows.calibrated_referable_probability=[0.1;0.886];
    rows.continuous_score=[0.39;2.19];
    rows.confidence=[0.85;0.412];
end

function identity=validIdentity
    digest=repmat('a',1,64);
    identity=struct('schema_version',1,'model_sha256',digest,'config_sha256',digest, ...
        'labels_sha256',digest,'mode','model','selection_sha256','', ...
        'preprocess_sha256',digest);
end

function identity=versionTwo(rows)
    identity=validIdentity; identity.schema_version=2;
    identity.source_sha256=repmat('c',1,64);
    for index=1:height(rows)
        identity.expected_cohort(index)=struct('id_code',char(rows.id_code(index)), ...
            'true_grade',rows.true_grade(index),'image_sha256',repmat('d',1,64));
    end
end

function writeChunk(root,name,rows,identity)
    folder=fullfile(root,name);
    if ~isfolder(folder), mkdir(folder); end
    writetable(rows,fullfile(folder,'per_image_results.csv'));
    fid=fopen(fullfile(folder,'run_identity.json'),'w');
    cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s',jsonencode(identity));
end
