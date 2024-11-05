# from tools.vqgan.extract_vq import main as extract_vq 
from tools.vqgan.inference import main as inference
from tools.llama.generate import main as generate



def get_task_list():
   return 0
def run_1(task_id: str, input_text: str, prompt_v: str):
  # ------------------------------------------------------------
  # prompt_v = 'data/spk1/t-01'

  result_path='results/' + task_id + '.wav'

  # ------------------------------------------------------------
  config_name='firefly_gan_vq'
  checkpoint_path='checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth'

  # python tools/vqgan/extract_vq.py data \
  #     --num-workers 1 --batch-size 16 \
  #     --config-name "firefly_gan_vq" \
  #     --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

  # 提前批量处理
  # 对原始音频文件 提取语义token
  # extract_vq(folder='data', num_workers=1, batch_size=16, checkpoint_path, config_name) # type: ignore


  # ------------------------------------------------------------
  # filelist = '' # type Path
  # 单个处理了
  # extract_vq(filelist==filelist, num_workers=1, batch_size=16, checkpoint_path=checkpoint_path, config_name='firefly_gan_vq')  # type: ignore

  # ------------------------------------------------------------
  # python tools/llama/generate.py \
  #     --text "要转换的文本" \
  #     --prompt-text "你的参考文本" \
  #     --prompt-tokens "paimon.npy" \
  #     --checkpoint-path "checkpoints/fish-speech-1.4" \
  #     --num-samples 1 \
  #     --compile



  with open(prompt_v + ".lab", "r", encoding="utf-8") as file:
      prompt_v_text = file.read()

  checkpoint_path = 'checkpoints/fish-speech-1.4'
  generate(text=input_text, prompt_text=prompt_v_text, prompt_tokens=prompt_v + '.npy', num_samples=1, compile=True, task_id=task_id, checkpoint_path=checkpoint_path)
  # type: ignore

  # ------------------------------------------------------------

  # python tools/vqgan/inference.py \
  #     -i "codes_0.npy" \
  #     -o "jojo-01.wav" \
  #     --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"


  inference(input_path=task_id + '_0.npy', output_path=result_path, checkpoint_path=checkpoint_path)
  # type: ignore

  # ------------------------------------------------------------
  # TODO 将生成的结果上传到一个地方


def main_1():
   input_text = '最近，长白山景区准备了接驳车！就是上汽大通V80！有了它，游客上山更方便。前几天，长白山景区和上汽大通合作，准备了好多V80作为接驳车！游客上山再也不用排队了。'
   task_list = [
      ['1105-jojo-01', 'data/jojo/001'],
      # ['1105-jojo-02', 'data/jojo/002'],
      # ['1105-yangge-01', 'data/yangge/003'],
      # ['1105-yangge-02', 'data/yangge/004'],
   ]

   for task in task_list:
      run_1(task[0], input_text, task[1])


main_1()