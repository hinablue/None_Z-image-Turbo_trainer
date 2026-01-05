<template>
  <el-dialog
    v-model="visible"
    title="分桶计算器"
    width="800px"
    class="bucket-dialog"
    @close="handleClose"
  >
    <div class="bucket-config">
      <el-form :inline="true" label-width="100px">
        <el-form-item label="Batch Size">
          <el-input-number v-model="config.batchSize" :min="1" :max="16" />
        </el-form-item>
        <el-form-item label="分辨率限制">
          <el-input-number v-model="config.resolutionLimit" :min="256" :max="2048" :step="64" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="calculate" :loading="calculating">
            计算分桶
          </el-button>
        </el-form-item>
      </el-form>
    </div>
    
    <div class="bucket-results" v-if="results.length > 0">
      <div class="bucket-summary">
        <div class="summary-item">
          <span class="label">总图片数</span>
          <span class="value">{{ summary.totalImages }}</span>
        </div>
        <div class="summary-item">
          <span class="label">桶数量</span>
          <span class="value">{{ results.length }}</span>
        </div>
        <div class="summary-item">
          <span class="label">总批次数</span>
          <span class="value">{{ summary.totalBatches }}</span>
        </div>
        <div class="summary-item">
          <span class="label">丢弃图片</span>
          <span class="value" :class="{ 'text-warning': summary.droppedImages > 0 }">
            {{ summary.droppedImages }}
          </span>
        </div>
      </div>
      
      <el-table :data="results" style="width: 100%" max-height="400">
        <el-table-column prop="resolution" label="分辨率" width="120">
          <template #default="{ row }">
            {{ row.width }}×{{ row.height }}
          </template>
        </el-table-column>
        <el-table-column prop="aspectRatio" label="宽高比" width="100">
          <template #default="{ row }">
            {{ row.aspectRatio.toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column prop="count" label="图片数" width="80" />
        <el-table-column prop="batches" label="批次数" width="80" />
        <el-table-column prop="dropped" label="丢弃" width="60">
          <template #default="{ row }">
            <span :class="{ 'text-warning': row.dropped > 0 }">{{ row.dropped }}</span>
          </template>
        </el-table-column>
        <el-table-column label="分布" min-width="200">
          <template #default="{ row }">
            <el-progress 
              :percentage="row.percentage" 
              :stroke-width="12"
              :show-text="false"
              :color="getBucketColor(row.aspectRatio)"
            />
          </template>
        </el-table-column>
      </el-table>
    </div>
    
    <div class="bucket-empty" v-else-if="!calculating">
      <el-icon :size="48"><Grid /></el-icon>
      <p>点击「计算分桶」查看数据集的分桶分布</p>
    </div>
    
    <template #footer>
      <el-button @click="handleClose">关闭</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { Grid } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

// Props
const props = defineProps<{
  modelValue: boolean
  datasetPath: string
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

// 本地状态
const visible = ref(props.modelValue)
const config = ref({
  batchSize: 4,
  resolutionLimit: 1536
})
const calculating = ref(false)
const results = ref<BucketInfo[]>([])

interface BucketInfo {
  width: number
  height: number
  aspectRatio: number
  count: number
  batches: number
  dropped: number
  percentage: number
}

// 同步 visible 与 modelValue
watch(() => props.modelValue, (val) => {
  visible.value = val
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

// 汇总统计
const summary = computed(() => {
  const totalImages = results.value.reduce((sum, b) => sum + b.count, 0)
  const totalBatches = results.value.reduce((sum, b) => sum + b.batches, 0)
  const droppedImages = results.value.reduce((sum, b) => sum + b.dropped, 0)
  return { totalImages, totalBatches, droppedImages }
})

// 根据宽高比返回颜色
function getBucketColor(aspectRatio: number): string {
  if (aspectRatio < 0.8) return '#67c23a' // 竖图 - 绿色
  if (aspectRatio > 1.2) return '#409eff' // 横图 - 蓝色
  return '#f0b429' // 方图 - 金色
}

// 计算分桶
async function calculate() {
  if (!props.datasetPath) return
  
  calculating.value = true
  results.value = []
  
  try {
    const response = await axios.post('/api/dataset/calculate-buckets', {
      path: props.datasetPath,
      batch_size: config.value.batchSize,
      resolution_limit: config.value.resolutionLimit
    })
    
    results.value = response.data.buckets
    
  } catch (error: any) {
    ElMessage.error('计算分桶失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    calculating.value = false
  }
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.bucket-config {
  margin-bottom: var(--space-md);
}

.bucket-summary {
  display: flex;
  gap: var(--space-lg);
  margin-bottom: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  
  .summary-item {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    
    .label {
      font-size: 12px;
      color: var(--text-muted);
    }
    
    .value {
      font-size: 20px;
      font-weight: bold;
      
      &.text-warning {
        color: var(--color-warning);
      }
    }
  }
}

.bucket-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-xl);
  color: var(--text-muted);
  
  p {
    margin-top: var(--space-md);
  }
}

.text-warning {
  color: var(--color-warning);
}
</style>
