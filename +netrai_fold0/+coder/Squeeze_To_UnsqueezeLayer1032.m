classdef Squeeze_To_UnsqueezeLayer1032 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = netrai_fold0.coder.Squeeze_To_UnsqueezeLayer1032(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = netrai_fold0.Squeeze_To_UnsqueezeLayer1032(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = Squeeze_To_UnsqueezeLayer1032(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_Unsqueeze_output_0'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = netrai_fold0.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [x_Unsqueeze_output_0] = predict(this, x_model_ref_head_Gem__)
            if isdlarray(x_model_ref_head_Gem__)
                x_model_ref_head_Gem_ = stripdims(x_model_ref_head_Gem__);
            else
                x_model_ref_head_Gem_ = x_model_ref_head_Gem__;
            end
            x_model_ref_head_GemNumDims = 2;
            x_model_ref_head_Gem = netrai_fold0.coder.ops.permuteInputVar(x_model_ref_head_Gem_, [2 1], 2);

            [x_Unsqueeze_output_0__, x_Unsqueeze_output_0NumDims__] = Squeeze_To_UnsqueezeGraph1096(this, x_model_ref_head_Gem, x_model_ref_head_GemNumDims, false);
            x_Unsqueeze_output_0_ = netrai_fold0.coder.ops.permuteOutputVar(x_Unsqueeze_output_0__, [2 1], 2);

            x_Unsqueeze_output_0 = dlarray(single(x_Unsqueeze_output_0_), 'CB');
        end

        function [x_Unsqueeze_output_0, x_Unsqueeze_output_0NumDims1097] = Squeeze_To_UnsqueezeGraph1096(this, x_model_ref_head_Gem, x_model_ref_head_GemNumDims, Training)

            % Execute the operators:
            % Squeeze:
            [x_model_Squeeze_outp, x_model_Squeeze_outpNumDims] = netrai_fold0.coder.ops.onnxSqueeze(x_model_ref_head_Gem, this.Vars.x_model_Constant_out, coder.const(x_model_ref_head_GemNumDims));

            % Unsqueeze:
            [shape1064, x_Unsqueeze_output_0NumDims] = netrai_fold0.coder.ops.prepareUnsqueezeArgs(x_model_Squeeze_outp, this.Vars.x_Constant_output_0, coder.const(x_model_Squeeze_outpNumDims));
            x_Unsqueeze_output_0 = reshape(x_model_Squeeze_outp, shape1064);

            % Set graph output arguments
            x_Unsqueeze_output_0NumDims1097 = coder.const(x_Unsqueeze_output_0NumDims);

        end

    end

end